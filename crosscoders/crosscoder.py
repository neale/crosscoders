import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
from transformer_lens import HookedTransformer


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16
}

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
        train_layers: List[Any],
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
        self.n_layers_base = model.cfg.n_layers
        self.n_layers_model = train_layers
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
        # x shape: (n_layers, batch_size, d_model)
        x_flat = x.reshape(-1, self.d_model)
        scores = F.linear(x_flat, self.encoder)  # (n_layers, batch_size * seq_len, dict_size)
        return scores.reshape(x.shape[0], x.shape[1], self.dict_size)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode dictionary elements back into activations."""
        # x shape: (n_layers, batch_size, dict_size)
        x_flat = x.reshape(-1, self.dict_size)
        decoded = F.linear(x_flat, self.decoder)  # (n_layers, batch_size * seq_len, d_model)
        return decoded.reshape(x.shape[0], x.shape[1], self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded