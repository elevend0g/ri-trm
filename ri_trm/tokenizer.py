"""
Real Tokenizer Integration for RI-TRM

Uses GPT-2 tokenizer for actual code generation and verification.
This replaces the placeholder tokenization that was preventing
the model from learning.
"""

import torch
from typing import List, Union, Optional
from transformers import GPT2Tokenizer, AutoTokenizer


class CodeTokenizer:
    """
    Real tokenizer for code generation

    Uses GPT-2/CodeGen tokenizer for actual token <-> code conversion.
    Handles padding, truncation, and special tokens properly.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 512,
        padding_side: str = "right"
    ):
        """
        Initialize tokenizer

        Args:
            model_name: HuggingFace model name (gpt2, Salesforce/codegen-350M-mono, etc.)
            max_length: Maximum sequence length
            padding_side: Which side to pad on ('right' or 'left')
        """
        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer
        if "codegen" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Set padding token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Configure padding
        self.tokenizer.padding_side = padding_side

        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = True
    ) -> Union[torch.Tensor, List[int]]:
        """
        Encode text to token IDs

        Args:
            text: Text to encode
            max_length: Override default max length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: Return torch.Tensor instead of list

        Returns:
            Token IDs as tensor [L] or list
        """
        max_len = max_length or self.max_length

        encoded = self.tokenizer.encode(
            text,
            max_length=max_len,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors=None  # We'll handle tensor conversion ourselves
        )

        if return_tensors:
            return torch.tensor(encoded, dtype=torch.long)
        return encoded

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip padding/eos/bos tokens
            clean_up_tokenization_spaces: Clean up spaces around punctuation

        Returns:
            Decoded text string
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        # Remove padding tokens manually for better control
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t != self.pad_token_id]

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> torch.Tensor:
        """
        Encode batch of texts

        Args:
            texts: List of texts to encode
            max_length: Override default max length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Token IDs tensor [B, L]
        """
        max_len = max_length or self.max_length

        encoded = self.tokenizer(
            texts,
            max_length=max_len,
            padding="max_length" if padding else True,
            truncation=truncation,
            return_tensors="pt"
        )

        return encoded["input_ids"]

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs

        Args:
            token_ids: Token IDs tensor [B, L]
            skip_special_tokens: Skip padding/eos/bos tokens

        Returns:
            List of decoded strings
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]

    def get_token_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get mask for non-padding tokens

        Args:
            token_ids: Token IDs tensor [B, L] or [L]

        Returns:
            Boolean mask [B, L] or [L] where True = valid token
        """
        return token_ids != self.pad_token_id

    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size


# Global tokenizer instance
_global_tokenizer: Optional[CodeTokenizer] = None


def get_tokenizer(
    model_name: str = "gpt2",
    max_length: int = 512,
    force_reload: bool = False
) -> CodeTokenizer:
    """
    Get global tokenizer instance (singleton pattern)

    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        force_reload: Force reload even if already initialized

    Returns:
        CodeTokenizer instance
    """
    global _global_tokenizer

    if _global_tokenizer is None or force_reload:
        _global_tokenizer = CodeTokenizer(
            model_name=model_name,
            max_length=max_length
        )

    return _global_tokenizer


def tokens_to_code(token_ids: Union[torch.Tensor, List[int]]) -> str:
    """
    Convenience function: Convert token IDs to code string

    Args:
        token_ids: Token IDs as tensor or list

    Returns:
        Decoded code string
    """
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids)


def code_to_tokens(code: str, max_length: int = 512) -> torch.Tensor:
    """
    Convenience function: Convert code string to token IDs

    Args:
        code: Code string
        max_length: Maximum sequence length

    Returns:
        Token IDs tensor [L]
    """
    tokenizer = get_tokenizer()
    return tokenizer.encode(code, max_length=max_length)
