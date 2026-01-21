"""
Unit tests for the TextChunker class.
"""

import pytest
from src.text_processing.chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker."""
    
    def test_initialization(self):
        """Test chunker initialization."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
    
    def test_invalid_overlap(self):
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=50, overlap=60)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
    
    def test_simple_chunking(self):
        """Test basic text chunking."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunker = TextChunker(chunk_size=30, overlap=10)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_number' in chunk for chunk in chunks)
        assert chunks[0]['chunk_number'] == 1
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunker = TextChunker(chunk_size=30, overlap=10)
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that some words from end of first chunk appear in second
            first_words = chunks[0]['text'].split()
            second_words = chunks[1]['text'].split()
            # Last words of first chunk should be in second chunk
            assert any(word in second_words for word in first_words[-10:])
    
    def test_chunk_statistics(self):
        """Test chunk statistics calculation."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunker = TextChunker(chunk_size=30, overlap=10)
        chunks = chunker.chunk_text(text)
        
        stats = chunker.get_chunk_statistics(chunks)
        
        assert stats['total_chunks'] == len(chunks)
        assert stats['chunk_size'] == 30
        assert stats['overlap'] == 10
        assert stats['total_words'] > 0
    
    def test_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = "word1 word2 word3"
        chunker = TextChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == text