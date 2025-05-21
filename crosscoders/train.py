import torch
import torch.nn.functional as F
import argparse
import json, yaml
from pathlib import Path
import logging
from .utils import *
from .crosscoder import CrossCoder
from .buffer import Buffer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_args():
    def parse_layers(val):
        return val if val == "all" else [int(x) for x in val.split(",")]
    parser = argparse.ArgumentParser(description="Train a CrossCoder on a HuggingFace model")

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
    parser.add_argument("--layers", type=parse_layers, default=None,
                       help='Layers to train on')
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
    logger.info("Updated config:")
    logger.info(json.dumps(cfg, indent=2))
    
    return cfg


class Trainer:
    """
    Trainer class for CrossCoder.
    
    Args:
        crosscoder: The CrossCoder model
        buffer: The Buffer for data
        lr: Learning rate
        l1_coeff: L1 regularization coefficient
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        num_tokens: Number of tokens to train on
        model_batch_size: Batch size for model forward passes
        log_every: Logging frequency
        save_every: Checkpoint saving frequency
        save_dir: Directory to save checkpoints
    """
    def __init__(
        self,
        crosscoder: CrossCoder,
        buffer: Buffer,
        lr: float = 2e-5,
        l1_coeff: float = 2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        num_tokens: int = int(4e8),
        model_batch_size: int = 32,
        log_every: int = 100,
        save_every: int = 100000,
        save_dir: str = "checkpoints"
    ):
        self.crosscoder = crosscoder
        self.buffer = buffer
        self.lr = lr
        self.l1_coeff = l1_coeff
        self.num_tokens = num_tokens
        self.model_batch_size = model_batch_size
        self.log_every = log_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        if isinstance(self.lr, str):
            self.lr = float(self.lr)
            
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=self.lr,
            betas=(beta1, beta2)
        )
        
        # Initialize loss tracking
        self.losses = []
        self.l1_losses = []
        self.total_tokens = 0
        
    def train(self) -> None:
        """Train the CrossCoder model."""
        logger.info("Starting training...")
        batch_idx = 0
        while self.total_tokens < self.num_tokens:
            # Get batch from buffer
            acts = self.buffer.get_batch() # [layers x]
            batch_idx += 1
            if acts is None:
                self.buffer.refresh()
                acts = self.buffer.get_batch()
            
            z = self.crosscoder.encode(acts)             # [layers, B*T, k]
            decoded = self.crosscoder.decode(z)          # same shape as `acts`
            recon_loss = F.mse_loss(decoded, acts)

            dec_norm_sum   = self.crosscoder.decoder.norm(dim=-1).sum(dim=0)   # (k,)
            latent_mean    = z.abs().mean(dim=(0,1))                           # (k,)
            sparse_loss    = (latent_mean * dec_norm_sum).mean()
            l1_loss        = self.l1_coeff * sparse_loss
            loss = recon_loss + l1_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.losses.append(recon_loss.item())
            self.l1_losses.append(l1_loss.item())
            self.total_tokens += acts.size(1)
            if batch_idx % self.log_every == 0:
                logger.info(
                    f"Tokens: {self.total_tokens}/{self.num_tokens}, "
                    f"L2 Loss: {recon_loss.item():.4f}, "
                    f"L1 Loss: {l1_loss.item():.4f}, "
                    f"Total Loss: {loss.item():.4f}"
                )
            # Anthropic: measure the eval loss as MSE + decoder norm-weighted L1 norm, summed across layers:
            # Save checkpoint
            if self.total_tokens % self.save_every == 0:
                self.save_checkpoint()
        
        # Save final checkpoint
        self.save_checkpoint()
        logger.info("Training complete!")
    
    def save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.crosscoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
            "l1_losses": self.l1_losses,
            "total_tokens": self.total_tokens
        }
        
        save_path = self.save_dir / f"checkpoint_{self.total_tokens}.pt"
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        

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
    
    # Init CLT
    if cfg['layers'] == 'all':
        cfg['layers'] = range(model.cfg.n_layers)
        
    crosscoder = CrossCoder(
        dict_size=cfg.get('dict_size'),
        site=cfg.get('site'),
        model=model,
        train_layers=cfg.get('layers'),
        device=cfg.get('device'),
        dtype=cfg.get('dtype'),        
    )
    # Init buffer
    buffer = Buffer(
        cfg=cfg,
        model=model,
        normalization_factors=norm_factors,
        data_source=cfg.get('data_source'),
        data_type=cfg.get('data_type'),
        text_column=cfg.get('text_column'),
        num_samples=cfg.get('num_samples'),
        save_tokens_path=cfg.get('save_tokens_path')        
    )
    buffer.refresh()  # collects activations
    
    # Initialize and train
    trainer = Trainer(
        crosscoder=crosscoder,
        buffer=buffer,
        lr=cfg.get('lr'),
        l1_coeff=cfg.get('l1_coeff'),
        num_tokens=cfg.get('num_tokens'),
        model_batch_size=cfg.get('model_batch_size'),
    )
    trainer.train()

if __name__ == "__main__":
    main()