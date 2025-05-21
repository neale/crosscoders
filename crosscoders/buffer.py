import torch
import einops
import tqdm
from pathlib import Path
from transformer_lens import HookedTransformer
from typing import Any, Optional, Union, Dict
import logging
from .data_loader import DataLoader
from .utils import verify_activation_site, DTYPES, calculate_normalization_factors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Buffer:
    """Data buffer for training the autoencoder.
        Holds a buffer of tokenized data as well as per-layer normalization factors
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: HookedTransformer,
        normalization_factors: Optional[torch.Tensor] = None,
        data_source: Optional[Union[str, Path]] = None,
        data_type: str = "hf",  # or "json" or "tokens"
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
        train_layers = cfg.get('layers')
        n_train_layers = len(train_layers)
        self.buffer = torch.zeros(
            (self.buffer_size, n_train_layers, model.cfg.d_model),
            dtype=DTYPES[cfg["enc_dtype"]],
            requires_grad=False,
            device=cfg["device"]
        )
        logger.info(f"Initializing empty buffer of shape: {self.buffer.shape}")
        
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
            
        # logger.info(f"Using normalization factors: {self.normalisation_factor}")
        
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
        
    def get_batch(self) -> torch.Tensor:
        """Return the next batch of activations from the buffer."""
        if not hasattr(self, 'pointer'):
            self.pointer = 0
        batch_size = self.cfg["batch_size"]
        if self.pointer + batch_size > self.buffer.shape[0]:
            self.refresh()
            self.pointer = 0
        batch = self.buffer[self.pointer:self.pointer + batch_size]
        self.pointer += batch_size
        # Rearrange to (n_layers, batch, d_model) for CrossCoder
        batch = einops.rearrange(batch, 'b n_layers d_model -> n_layers b d_model')
        batch = batch.to(DTYPES[self.cfg['enc_dtype']])
        return batch
        
    def drop_out_of_site(self, acts, cache):
        target_layers = self.cfg.get('layers')
        layer_keys = [f'.{l}.' for l in target_layers]
        cache_keys = list(cache.keys())
        act_idxs = []
        for i in range(len(acts)):
            for layer in layer_keys:
                if layer in cache_keys[i]:
                    act_idxs.append(i)
                    break
        return acts[act_idxs]
    
    @torch.no_grad()
    def refresh(self, drop_extra_sites=True) -> None:
        """Refresh the buffer with new activations."""
        self.pointer = 0
        logger.info("Refreshing the buffer!")
        
        with torch.autocast(self.cfg["device"], DTYPES[self.cfg["enc_dtype"]]):
            num_batches = self.buffer_batches if self.first else self.buffer_batches // 2
            self.first = False
            
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                # Use loaded tokens
                batch_indices = torch.randint(
                    0, len(self.tokens),
                    (self.cfg["model_batch_size"],),
                    device=self.cfg["device"]
                )
                tokens = self.tokens[batch_indices]
                # Get activations for all layers (can't see how to stack acts without doing all layers)
                target_layers = self.cfg.get('layers')
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda x: x.endswith(self.cfg["site"])
                    #names_filter=lambda x: any(str(l) in x for l in target_layers) and x.endswith(self.cfg["site"])
                )

                # Stack and normalize activations
                acts = cache.stack_activation(self.cfg["site"])
                if drop_extra_sites:
                    acts = self.drop_out_of_site(acts, cache)
                acts = acts[:, :, 1:, :]  # Drop BOS
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )  
                # [batch * seq_len, n_layers_full, d_model_site]
                if self.normalize:
                    acts = acts / self.normalisation_factor[None, :, None] 
                
                if acts.shape[0] + self.pointer >= self.buffer.shape[0]:
                    logger.info("Buffer Overrun, resizing...")
                    new_len = self.pointer + acts.shape[0] + self.buffer.shape[0]
                    new_storage = torch.zeros(new_len, *self.buffer.shape[1:]).to(self.buffer.device)
                    self.buffer = torch.cat((self.buffer, new_storage), dim=0)
                self.buffer[self.pointer:self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
        
        logger.info(f"Refreshed the buffer to size: {self.buffer.shape}")

