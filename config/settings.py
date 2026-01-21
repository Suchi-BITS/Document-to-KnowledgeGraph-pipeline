"""
Configuration settings for the Knowledge Graph Builder.
Loads settings from environment variables with sensible defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Central configuration class for all application settings."""
    
    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "150"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    
    # Visualization Configuration
    GRAPH_LAYOUT: str = os.getenv("GRAPH_LAYOUT", "cose")
    ANIMATE_LAYOUT: bool = os.getenv("ANIMATE_LAYOUT", "true").lower() == "true"
    
    # Layout Parameters for COSE algorithm
    LAYOUT_NODE_REPULSION: int = 4000
    LAYOUT_NODE_OVERLAP: int = 40
    LAYOUT_IDEAL_EDGE_LENGTH: int = 120
    LAYOUT_EDGE_ELASTICITY: int = 150
    LAYOUT_NESTING_FACTOR: int = 5
    LAYOUT_GRAVITY: int = 100
    LAYOUT_NUM_ITER: int = 1500
    LAYOUT_INITIAL_TEMP: int = 200
    LAYOUT_COOLING_FACTOR: float = 0.95
    LAYOUT_MIN_TEMP: float = 1.0
    
    # Node styling
    NODE_MIN_SIZE: int = 15
    NODE_MAX_SIZE_FACTOR: int = 50
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE and cls.CHUNK_SIZE > 0:
            raise ValueError(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})."
            )
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration as a dictionary."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "base_url": cls.OPENAI_API_BASE,
            "model": cls.LLM_MODEL_NAME,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
        }
    
    @classmethod
    def get_chunk_config(cls) -> dict:
        """Get text chunking configuration as a dictionary."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "overlap": cls.CHUNK_OVERLAP,
        }
    
    @classmethod
    def get_layout_config(cls) -> dict:
        """Get graph layout configuration as a dictionary."""
        return {
            "name": cls.GRAPH_LAYOUT,
            "animate": cls.ANIMATE_LAYOUT,
            "nodeRepulsion": cls.LAYOUT_NODE_REPULSION,
            "nodeOverlap": cls.LAYOUT_NODE_OVERLAP,
            "idealEdgeLength": cls.LAYOUT_IDEAL_EDGE_LENGTH,
            "edgeElasticity": cls.LAYOUT_EDGE_ELASTICITY,
            "nestingFactor": cls.LAYOUT_NESTING_FACTOR,
            "gravity": cls.LAYOUT_GRAVITY,
            "numIter": cls.LAYOUT_NUM_ITER,
            "initialTemp": cls.LAYOUT_INITIAL_TEMP,
            "coolingFactor": cls.LAYOUT_COOLING_FACTOR,
            "minTemp": cls.LAYOUT_MIN_TEMP,
        }


# Create a global settings instance
settings = Settings()