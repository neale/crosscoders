import torch
import argparse
import json, yaml
from typing import Optional, Dict, Any, Union
from pathlib import Path
from .utils_new import (
    CrossCoder,
    Trainer,
    Buffer,
    get_default_cfg,
    arg_parse_update_cfg,
    post_init_cfg,
    DTYPES,
    verify_activation_site,
    calculate_normalization_factors
)
from transformer_lens import HookedTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(description="Train a CrossCoder on a HuggingFace model")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    # Model arguments
    parser.add_argument("--model_name", type=str,
                       help="HuggingFace model identifier")
    parser.add_argument("--device", type=str,
                       help="Device to run on (e.g. cuda:0, cpu)")
    parser.add_argument("--dtype", type=str,
                       help="Data type to use (bf16, fp32, fp16)")
    parser.add_argument("--site", type=str,
                       help="Activation site to use (default: resid_post)")
    
    # Data arguments
    parser.add_argument("--data_source", type=str,
                       help="Path to data file or HuggingFace dataset name")
    parser.add_argument("--data_type", type=str,
                       choices=["hf", "json", "tokens"],
                       help="Type of data source")
    parser.add_argument("--text_column", type=str,
                       help="Column name containing text in dataset")
    parser.add_argument("--num_samples", type=int,
                       help="Number of samples to use from dataset")
    parser.add_argument("--save_tokens", type=str,
                       help="Path to save tokenized data")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int,
                       help="Training batch size")
    parser.add_argument("--buffer_mult", type=int,
                       help="Buffer multiplier")
    parser.add_argument("--lr", type=float,
                       help="Learning rate")
    parser.add_argument("--l1_coeff", type=float,
                       help="L1 regularization coefficient")
    parser.add_argument("--dict_size", type=int,
                       help="Dictionary size")
    parser.add_argument("--num_tokens", type=int,
                       help="Total number of tokens to train on")
    parser.add_argument("--seq_len", type=int,
                       help="Sequence length")
    parser.add_argument("--model_batch_size", type=int,
                       help="Model batch size")
    
    # Logging and saving arguments
    parser.add_argument("--log_every", type=int,
                       help="Log every N steps")
    parser.add_argument("--save_every", type=int,
                       help="Save every N steps")
    parser.add_argument("--save_dir", type=str,
                       help="Directory to save checkpoints")
    
    # Normalization arguments
    parser.add_argument("--norm_factors_path", type=str,
                       help="Path to normalization factors file (JSON)")
    parser.add_argument("--save_norm_factors", type=str,
                       help="Path to save calculated normalization factors (JSON)")
    
    args = parser.parse_args()
    cfg = get_default_cfg()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            cfg.update(yaml_cfg)
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            cfg[key] = value
    
    post_init_cfg(cfg)
    return cfg


def load_model(model_name: str, device: str = "cuda:0", dtype: str = "bf16") -> HookedTransformer:
    """
    Load a model from HuggingFace using transformer_lens with proper error handling.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Data type to use (bf16, fp32, fp16)
        
    Returns:
        HookedTransformer instance
        
    Raises:
        ValueError: If model loading fails or dtype is invalid
    """
    if dtype not in DTYPES:
        raise ValueError(f"Invalid dtype {dtype}. Must be one of {list(DTYPES.keys())}")
        
    try:
        logger.info(f"Loading model {model_name} with dtype {dtype} on {device}")
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=DTYPES[dtype]
        )
        logger.info(f"Successfully loaded model with {model.cfg.n_layers} layers")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name}: {str(e)}")

def load_normalization_factors(path: Optional[str], model: HookedTransformer, cfg: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Load normalization factors from a file or calculate them if not provided.
    
    Args:
        path: Path to normalization factors file (JSON)
        model: The HookedTransformer model
        cfg: Configuration dictionary
        
    Returns:
        Optional[torch.Tensor]: Normalization factors if loaded or calculated
    """
    if path is None:
        return None
        
    try:
        with open(path, 'r') as f:
            factors_dict = json.load(f)
            
        # Check if factors are for the correct model and site
        if (factors_dict.get('model_name') != cfg['model_name'] or
            factors_dict.get('site') != cfg['site']):
            logger.warning(
                f"Normalization factors file is for model {factors_dict.get('model_name')} "
                f"and site {factors_dict.get('site')}, but current model is {cfg['model_name']} "
                f"and site is {cfg['site']}. Recalculating factors."
            )
            return None
            
        factors = torch.tensor(
            factors_dict['factors'],
            device=cfg['device'],
            dtype=DTYPES[cfg['enc_dtype']]
        )
        
        if len(factors) != model.cfg.n_layers:
            logger.warning(
                f"Normalization factors file has {len(factors)} layers, "
                f"but model has {model.cfg.n_layers} layers. Recalculating factors."
            )
            return None
            
        logger.info("Successfully loaded normalization factors from file")
        return factors
        
    except Exception as e:
        logger.warning(f"Failed to load normalization factors from file: {e}")
        return None

def save_normalization_factors(path: str, factors: torch.Tensor, cfg: Dict[str, Any]) -> None:
    """
    Save normalization factors to a file.
    
    Args:
        path: Path to save factors to
        factors: Normalization factors tensor
        cfg: Configuration dictionary
    """
    factors_dict = {
        'model_name': cfg['model_name'],
        'site': cfg['site'],
        'factors': factors.cpu().tolist()
    }
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(factors_dict, f, indent=2)
        
    logger.info(f"Saved normalization factors to {path}")

def main():
    cfg = load_args()
    
    try:
        logger.info("Trying to load model...")
        model = load_model(cfg["model_name"], cfg["device"], cfg["enc_dtype"])
    except ValueError as e:
        logger.error(e)
        return
    
    try:
        logger.info("Verifying Crosscoder training target layer exists...")
        verify_activation_site(model, cfg["site"])
    except ValueError as e:
        logger.error(f"Activation site verification failed: {e}")
        return
    
    # Load or calculate normalization factors
    norm_factors = load_normalization_factors(cfg.get("norm_factors_path"), model, cfg)
    
    # If no factors were loaded, calculate them
    if norm_factors is None:
        norm_factors = calculate_normalization_factors(
            model=model,
            site=cfg["site"],
            num_samples=1000,
            batch_size=cfg["model_batch_size"],
            seq_len=cfg["seq_len"],
            device=cfg["device"],
            dtype=cfg["enc_dtype"]
        )
        
        # Save factors if requested
        if cfg.get("save_norm_factors"):
            save_normalization_factors(cfg["save_norm_factors"], norm_factors, cfg)
    
    # Initialize and train
    trainer = Trainer(
        cfg,
        model,
        norm_factors,
        data_source=cfg["data_source"],
        data_type=cfg["data_type"],
        dataset_name=cfg["data_source"] if cfg["data_type"] == "hf" else None,
        text_column=cfg.get("text_column", "text"),
        num_samples=cfg.get("num_samples"),
        save_tokens_path=cfg.get("save_tokens")
    )
    trainer.train()

if __name__ == "__main__":
    main()