import torch
import numpy as np
import random
import tqdm
import json
import pprint
from pathlib import Path
from typing import Dict, Any, Union, Optional
from transformer_lens import HookedTransformer
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data types supported by the model
DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16
}

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_default_cfg() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        "seed": 51,
        "batch_size": 2048,
        "buffer_mult": 512,
        "lr": 2e-5,
        "num_tokens": int(4e8),
        "l1_coeff": 2,
        "beta1": 0.9,
        "beta2": 0.999,
        "dict_size": 2**16,
        "seq_len": 1024,
        "enc_dtype": "bf16",
        "model_name": "gpt2-small",
        "site": "resid_post",
        "device": "cuda:0",
        "model_batch_size": 32,
        "log_every": 100,
        "save_every": 100000,
        "dec_init_norm": 0.005,
        "save_dir": "checkpoints",  # Default save directory
    }

def post_init_cfg(cfg: Dict[str, Any]) -> None:
    """Post-initialization configuration updates."""
    cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['site']}"
    
    # Create save directory if it doesn't exist
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg["save_dir"] = str(save_dir)
    
    logger.info("Final configuration:")
    logger.info(pprint.pformat(cfg))

# Set random seeds for reproducibility
def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_grad_enabled(True)

def verify_activation_site(model: HookedTransformer, site: str = "resid_post") -> bool:
    """
    Verify that cache.stack_activation works correctly for the given site.
    
    Args:
        model: The HookedTransformer model
        site: The activation site to verify (default: "resid_post")
        
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    try:
        # Create a small test input
        test_input = torch.randint(
            0, model.cfg.d_vocab,
            (1, 10),  # Small batch and sequence length for testing
            device=model.cfg.device
        )
        
        # Run model and get cache
        _, cache = model.run_with_cache(
            test_input,
            names_filter=lambda x: x.endswith(site)
        )
        
        # Try to stack activations
        try:
            stacked = cache.stack_activation(site)
            logger.info(f"Successfully stacked activations for {site} into shape: {stacked.shape}")
            
            # Verify the shape makes sense
            expected_shape = (model.cfg.n_layers, 1, 10, model.cfg.d_model)
            if stacked.shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for stacked activations. "
                    f"Got {stacked.shape}, expected {expected_shape}"
                )
            
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to stack activations for {site}: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Failed to verify activation site {site}: {str(e)}")

def calculate_normalization_factors(
    model: HookedTransformer,
    site: str,
    num_samples: int = 1000,
    batch_size: int = 32,
    seq_len: int = 1024,
    device: str = "cuda:0",
    dtype: str = "bf16"
) -> torch.Tensor:
    """
    Calculate normalization factors for each layer by running the model on sample data.
    
    Args:
        model: The HookedTransformer model
        site: The activation site to calculate factors for
        num_samples: Number of samples to use for calculation
        batch_size: Batch size for each forward pass
        seq_len: Sequence length for each sample
        device: Device to run on
        dtype: Data type to use
        
    Returns:
        torch.Tensor: Normalization factors for each layer
    """
    logger.info(f"Calculating normalization factors for {site} using {num_samples} samples...")
    
    # Initialize factors tensor
    factors = torch.zeros(model.cfg.n_layers, device=device)
    total_samples = 0
    
    # Calculate in batches to be memory efficient
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad(), torch.autocast(device, DTYPES[dtype]):
        for batch_idx in tqdm.trange(num_batches):
            # Calculate actual batch size for this iteration
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            if current_batch_size <= 0:
                break
                
            # Generate random input
            tokens = torch.randint(
                0, model.cfg.d_vocab,
                (current_batch_size, seq_len),
                device=device
            )
            
            # Get activations
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda x: x.endswith(site)
            )
            
            # Stack activations and calculate norms
            acts = cache.stack_activation(site)
            # Drop BOS token
            acts = acts[:, :, 1:, :]
            
            # Calculate norms for each layer
            layer_norms = acts.norm(dim=-1).mean(dim=(1, 2))
            factors += layer_norms * current_batch_size
            total_samples += current_batch_size
            
    # Average over all samples
    factors /= total_samples
    
    # logger.info(f"Calculated normalization factors: {factors}")
    return factors

def load_model(model_name: str, device: str = "cuda:0", dtype: str = "bf16") -> HookedTransformer:
    """
    Load a model from HuggingFace using transformer_lens
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Data type to use (bf16, fp32, fp16)
        
    Returns:
        HookedTransformer instance
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
        print ('Model Architecture:')
        print (model)
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
        return factors[cfg['layers']]
        
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