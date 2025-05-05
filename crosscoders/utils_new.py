import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import tqdm
import pprint
import einops
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
import huggingface_hub
import argparse
from typing import NamedTuple, Dict, Any, Optional, Union
from datasets import load_dataset
from transformer_lens import HookedTransformer, ActivationCache
import logging
import yaml
from pathlib import Path
from .data_loader import DataLoader

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

def arg_parse_update_cfg(default_cfg: Dict[str, Any], config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Update configuration with command line arguments and YAML config.
    
    Args:
        default_cfg: Default configuration dictionary
        config_path: Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary
    """ 
    cfg = dict(default_cfg)
    
    # Load YAML config if provided
    if config_path:
        yaml_config = load_yaml_config(config_path)
        cfg.update(yaml_config)
    
    parser = argparse.ArgumentParser()
    
    for key, value in default_cfg.items():
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
            
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    
    logger.info("Updated config:")
    logger.info(json.dumps(cfg, indent=2))
    return cfg

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
        
    Raises:
        ValueError: If verification fails with details about the error
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
            logger.info(f"Successfully stacked activations for {site}")
            logger.info(f"Stacked shape: {stacked.shape}")
            
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
    
    logger.info(f"Calculated normalization factors: {factors}")
    return factors


class Buffer:
    """Data buffer for training the autoencoder."""
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: HookedTransformer,
        normalization_factors: Optional[torch.Tensor] = None,
        data_source: Optional[Union[str, Path]] = None,
        data_type: str = "hf",  # or "json" or "tokens"
        dataset_name: Optional[str] = None,
        text_column: str = "text",
        num_samples: Optional[int] = None,
        save_tokens_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the buffer.
        
        Args:
            cfg: Configuration dictionary
            model: The HookedTransformer model
            normalization_factors: Optional normalization factors
            data_source: Path to data file or HuggingFace dataset name
            data_type: Type of data source ("hf", "json", or "tokens")
            dataset_name: Name of HuggingFace dataset (if data_type is "hf")
            text_column: Column name containing text in dataset
            num_samples: Number of samples to use
            save_tokens_path: Path to save tokenized data
        """
        if not data_source:
            raise ValueError("data_source must be provided")
            
        if data_type not in ["hf", "json", "tokens"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be one of: hf, json, tokens")
            
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        
        # Verify activation site works before proceeding
        try:
            verify_activation_site(model, cfg["site"])
        except ValueError as e:
            logger.error(f"Activation site verification failed: {e}")
            raise
        
        self.buffer = torch.zeros(
            (self.buffer_size, model.cfg.n_layers, model.cfg.d_model),
            dtype=DTYPES[cfg["enc_dtype"]],
            requires_grad=False,
            device=cfg["device"]
        )
        
        self.model = model
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        
        # Handle normalization factors
        if normalization_factors is not None:
            logger.info("Using provided normalization factors")
            self.normalisation_factor = normalization_factors.to(cfg["device"])
        else:
            logger.info("No normalization factors provided, calculating them...")
            self.normalisation_factor = calculate_normalization_factors(
                model=model,
                site=cfg["site"],
                num_samples=1000,
                batch_size=cfg["model_batch_size"],
                seq_len=cfg["seq_len"],
                device=cfg["device"],
                dtype=cfg["enc_dtype"]
            )
            
        # Verify normalization factors
        if torch.any(self.normalisation_factor <= 0):
            raise ValueError("Normalization factors must be positive")
            
        logger.info(f"Using normalization factors: {self.normalisation_factor}")
        
        # Initialize data loader and load data
        self.data_loader = DataLoader(
            model_name=cfg["model_name"],
            max_length=cfg["seq_len"],
            device=cfg["device"],
            dtype=cfg["enc_dtype"]
        )
        
        if data_type == "tokens":
            self.tokens = self.data_loader.load_tokens(data_source)
        else:
            if data_type == "hf":
                self.tokens = self.data_loader.load_huggingface(
                    dataset_name=data_source,
                    text_column=text_column,
                    num_samples=num_samples
                )
            elif data_type == "json":
                self.tokens = self.data_loader.load_json(data_source)
            else:
                raise ValueError(f"Invalid data_type: {data_type}")
                
            # Save tokenized data if requested
            if save_tokens_path:
                self.data_loader.save_tokens(self.tokens, save_tokens_path)
                
        logger.info(f"Loaded {len(self.tokens)} tokenized sequences")

            
    @torch.no_grad()
    def refresh(self) -> None:
        """Refresh the buffer with new activations."""
        self.pointer = 0
        logger.info("Refreshing the buffer!")
        
        with torch.autocast(self.cfg["device"], DTYPES[self.cfg["enc_dtype"]]):
            num_batches = self.buffer_batches if self.first else self.buffer_batches // 2
            self.first = False
            
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                if self.tokens is not None:
                    # Use loaded tokens
                    batch_indices = torch.randint(
                        0, len(self.tokens),
                        (self.cfg["model_batch_size"],),
                        device=self.cfg["device"]
                    )
                    tokens = self.tokens[batch_indices]
                else:
                    # Generate random tokens
                    tokens = torch.randint(
                        0, self.model.cfg.d_vocab,
                        (self.cfg["model_batch_size"], self.cfg["seq_len"]),
                        device=self.cfg["device"]
                    )
                
                # Get activations
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda x: x.endswith(self.cfg["site"])
                )
                
                # Stack and normalize activations
                acts = cache.stack_activation(self.cfg["site"])
                acts = acts[:, :, 1:, :]  # Drop BOS
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )
                
                if self.normalize:
                    acts = acts / self.normalisation_factor[None, :, None]
                
                self.buffer[self.pointer:self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                
        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]


class CrossCoder(nn.Module):
    """
    CrossCoder model that learns to encode and decode activations.
    
    Args:
        dict_size: Size of the dictionary
        site: Activation site to use
        model: The HookedTransformer model
        device: Device to run on
        dtype: Data type to use
        dec_init_norm: Initial norm for decoder weights
    """
    def __init__(
        self,
        dict_size: int,
        site: str,
        model: HookedTransformer,
        device: str = "cuda:0",
        dtype: str = "bf16",
        dec_init_norm: float = 0.005
    ):
        super().__init__()
        self.dict_size = dict_size
        self.site = site
        self.model = model
        self.device = device
        self.dtype = DTYPES[dtype]
        
        # Get model dimensions
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        
        # Initialize encoder and decoder
        self.encoder = nn.Parameter(torch.randn(dict_size, self.d_model, device=device))
        self.decoder = nn.Parameter(torch.randn(self.d_model, dict_size, device=device))
        
        # Initialize decoder with small norm
        with torch.no_grad():
            self.decoder.data *= dec_init_norm / self.decoder.data.norm(dim=0, keepdim=True)
        
        # Move to device and dtype
        self.to(device=device, dtype=self.dtype)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into dictionary elements."""
        # x shape: (n_layers, batch_size, seq_len, d_model)
        x_flat = x.reshape(-1, self.d_model)
        scores = F.linear(x_flat, self.encoder)  # (n_layers * batch_size * seq_len, dict_size)
        return scores.reshape(x.shape[0], x.shape[1], x.shape[2], self.dict_size)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode dictionary elements back into activations."""
        # x shape: (n_layers, batch_size, seq_len, dict_size)
        x_flat = x.reshape(-1, self.dict_size)
        decoded = F.linear(x_flat, self.decoder)  # (n_layers * batch_size * seq_len, d_model)
        return decoded.reshape(x.shape[0], x.shape[1], x.shape[2], self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


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
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        
        # Initialize loss tracking
        self.losses = []
        self.l1_losses = []
        self.total_tokens = 0
        
    def train(self) -> None:
        """Train the CrossCoder model."""
        logger.info("Starting training...")
        
        while self.total_tokens < self.num_tokens:
            # Get batch from buffer
            acts = self.buffer.next()
            if acts is None:
                self.buffer.refresh()
                acts = self.buffer.next()
            
            # Forward pass
            decoded = self.crosscoder(acts)
            
            # Calculate losses
            l2_loss = F.mse_loss(decoded, acts)
            l1_loss = self.l1_coeff * self.crosscoder.encoder.abs().mean()
            loss = l2_loss + l1_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update tracking
            self.losses.append(l2_loss.item())
            self.l1_losses.append(l1_loss.item())
            self.total_tokens += acts.shape[1] * acts.shape[2]
            
            # Logging
            if self.total_tokens % self.log_every == 0:
                logger.info(
                    f"Tokens: {self.total_tokens}/{self.num_tokens}, "
                    f"L2 Loss: {l2_loss.item():.4f}, "
                    f"L1 Loss: {l1_loss.item():.4f}"
                )
            
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