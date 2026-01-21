"""
Text chunking utilities for splitting large texts into manageable pieces.
Implements word-based chunking with overlap to preserve context.
"""

from typing import List, Dict
from config.settings import settings


class TextChunker:
    """Handles splitting text into chunks with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Number of words per chunk (defaults to settings)
            overlap: Number of words to overlap between chunks (defaults to settings)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        
        # Validate configuration
        if self.overlap >= self.chunk_size and self.chunk_size > 0:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than "
                f"chunk size ({self.chunk_size})."
            )
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of dictionaries containing chunk information:
            - text: The chunk text
            - chunk_number: Sequential chunk number (1-indexed)
            - start_word: Starting word index in original text
            - end_word: Ending word index in original text
        """
        # Split text into words
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return []
        
        chunks = []
        start_index = 0
        chunk_number = 1
        
        while start_index < total_words:
            # Calculate end index for this chunk
            end_index = min(start_index + self.chunk_size, total_words)
            
            # Extract chunk text
            chunk_text = " ".join(words[start_index:end_index])
            
            # Store chunk information
            chunks.append({
                "text": chunk_text,
                "chunk_number": chunk_number,
                "start_word": start_index,
                "end_word": end_index - 1,
                "word_count": end_index - start_index,
            })
            
            # Calculate start of next chunk
            next_start_index = start_index + self.chunk_size - self.overlap
            
            # Ensure we make progress (safety check)
            if next_start_index <= start_index:
                if end_index == total_words:
                    break  # Already processed the last part
                next_start_index = start_index + 1
            
            start_index = next_start_index
            chunk_number += 1
            
            # Safety break to prevent infinite loops
            if chunk_number > total_words:
                break
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_words": 0,
                "avg_words_per_chunk": 0,
                "min_words": 0,
                "max_words": 0,
            }
        
        word_counts = [chunk["word_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_words": sum(word_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }