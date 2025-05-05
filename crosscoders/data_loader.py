import json
import torch
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset, Dataset
import logging
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and tokenizing data from various sources."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 1024,
        device: str = "cuda:0",
        dtype: str = "bf16"
    ):
        """
        Initialize the data loader.
        
        Args:
            model_name: Name of the model to use for tokenization
            max_length: Maximum sequence length
            device: Device to load tensors on
            dtype: Data type to use
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_json(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and tokenize data from a JSON file.
        
        Expected JSON format:
        {
            "texts": [
                "text1",
                "text2",
                ...
            ]
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            torch.Tensor: Tokenized data
        """
        logger.info(f"Loading data from JSON file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'texts' not in data:
            raise ValueError("JSON file must contain a 'texts' key with a list of strings")
            
        return self._tokenize_texts(data['texts'])
        
    def load_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load and tokenize data from HuggingFace datasets.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to use
            text_column: Name of the column containing text
            num_samples: Number of samples to use (None for all)
            
        Returns:
            torch.Tensor: Tokenized data
        """
        logger.info(f"Loading dataset {dataset_name} from HuggingFace")
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        texts = dataset[text_column]
        return self._tokenize_texts(texts)
        
    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenize a list of texts and return as a tensor.
        
        Args:
            texts: List of text strings
            
        Returns:
            torch.Tensor: Tokenized data
        """
        logger.info(f"Tokenizing {len(texts)} texts")
        
        # Tokenize in batches to handle memory efficiently
        batch_size = 1000
        all_tokens = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            tokenized = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device and convert dtype
            tokens = tokenized['input_ids'].to(device=self.device, dtype=self.dtype)
            all_tokens.append(tokens)
            
        # Concatenate all batches
        tokens = torch.cat(all_tokens, dim=0)
        logger.info(f"Tokenized data shape: {tokens.shape}")
        
        return tokens
        
    def save_tokens(self, tokens: torch.Tensor, file_path: Union[str, Path]) -> None:
        """
        Save tokenized data to a file.
        
        Args:
            tokens: Tokenized data tensor
            file_path: Path to save to
        """
        logger.info(f"Saving tokenized data to {file_path}")
        torch.save(tokens, file_path)
        
    def load_tokens(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Load previously tokenized data from a file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            torch.Tensor: Tokenized data
        """
        logger.info(f"Loading tokenized data from {file_path}")
        return torch.load(file_path, map_location=self.device).to(dtype=self.dtype)