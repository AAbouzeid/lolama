"""Generation configuration for text generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Consolidates all generation parameters into a single object.
    
    Usage:
        config = GenerationConfig(max_new_tokens=100, temperature=0.7)
        output = model.generate(input_ids, config)
        
        # Or use defaults
        output = model.generate(input_ids, GenerationConfig())
    """
    
    # Length
    max_new_tokens: int = 50
    
    # Sampling
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    
    # Stopping
    eos_token_id: int | None = None
    
    # Penalties
    repetition_penalty: float = 1.0
    
    # Batch generation
    pad_token_id: int = 0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
    
    @classmethod
    def greedy(cls, max_new_tokens: int = 50, eos_token_id: int | None = None) -> GenerationConfig:
        """Create a greedy decoding config (deterministic)."""
        return cls(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )
    
    @classmethod
    def sampling(
        cls,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> GenerationConfig:
        """Create a typical sampling config."""
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_token_id,
        )
