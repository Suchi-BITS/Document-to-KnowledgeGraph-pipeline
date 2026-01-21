"""
Text normalization utilities for cleaning and standardizing extracted data.
Handles lowercase conversion, whitespace normalization, and deduplication.
"""

import re
from typing import List, Dict, Set, Tuple


class TripleNormalizer:
    """Normalizes and deduplicates extracted SPO triples."""
    
    def __init__(self):
        """Initialize the normalizer."""
        self.seen_triples: Set[Tuple[str, str, str]] = set()
        self.empty_removed_count = 0
        self.duplicates_removed_count = 0
    
    def normalize_triple(self, triple: Dict[str, any]) -> Dict[str, str]:
        """
        Normalize a single triple.
        
        Args:
            triple: Dictionary with 'subject', 'predicate', 'object' keys
            
        Returns:
            Normalized triple dictionary or None if invalid
        """
        subject_raw = triple.get('subject')
        predicate_raw = triple.get('predicate')
        object_raw = triple.get('object')
        
        # Check if all components are strings
        if not all(isinstance(val, str) for val in [subject_raw, predicate_raw, object_raw]):
            return None
        
        # Normalize: lowercase and trim whitespace
        normalized_sub = subject_raw.strip().lower()
        normalized_pred = re.sub(r'\s+', ' ', predicate_raw.strip().lower()).strip()
        normalized_obj = object_raw.strip().lower()
        
        # Check for empty components
        if not all([normalized_sub, normalized_pred, normalized_obj]):
            return None
        
        return {
            'subject': normalized_sub,
            'predicate': normalized_pred,
            'object': normalized_obj,
        }
    
    def normalize_and_deduplicate(
        self,
        triples: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Normalize and remove duplicate triples.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            List of normalized, unique triples with source_chunk information
        """
        normalized_triples = []
        self.seen_triples = set()
        self.empty_removed_count = 0
        self.duplicates_removed_count = 0
        
        for triple in triples:
            # Preserve source chunk information
            chunk_num = triple.get('chunk', 'unknown')
            
            # Normalize the triple
            normalized = self.normalize_triple(triple)
            
            if normalized is None:
                self.empty_removed_count += 1
                continue
            
            # Create identifier for deduplication
            triple_identifier = (
                normalized['subject'],
                normalized['predicate'],
                normalized['object']
            )
            
            # Check for duplicates
            if triple_identifier in self.seen_triples:
                self.duplicates_removed_count += 1
                continue
            
            # Add to results
            normalized['source_chunk'] = chunk_num
            normalized_triples.append(normalized)
            self.seen_triples.add(triple_identifier)
        
        return normalized_triples
    
    def get_statistics(self, original_count: int) -> Dict[str, int]:
        """
        Get normalization and deduplication statistics.
        
        Args:
            original_count: Number of original triples before processing
            
        Returns:
            Dictionary with statistics
        """
        final_count = original_count - self.empty_removed_count - self.duplicates_removed_count
        
        return {
            'original_count': original_count,
            'empty_removed': self.empty_removed_count,
            'duplicates_removed': self.duplicates_removed_count,
            'final_count': final_count,
        }
    
    def reset(self):
        """Reset the normalizer state."""
        self.seen_triples.clear()
        self.empty_removed_count = 0
        self.duplicates_removed_count = 0