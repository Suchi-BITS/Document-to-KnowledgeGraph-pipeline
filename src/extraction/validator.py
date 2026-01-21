"""
Validation module for extracted SPO triples.
Ensures triples have correct structure and valid data types.
"""

from typing import List, Dict, Any


class TripleValidator:
    """Validates the structure and content of extracted SPO triples."""
    
    def __init__(self):
        """Initialize the validator."""
        self.required_keys = {'subject', 'predicate', 'object'}
    
    def validate_triple(self, triple: Any) -> bool:
        """
        Validate a single triple.
        
        Args:
            triple: Potential triple to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if it's a dictionary
        if not isinstance(triple, dict):
            return False
        
        # Check if all required keys are present
        if not all(key in triple for key in self.required_keys):
            return False
        
        # Check if all values are strings
        if not all(isinstance(triple[key], str) for key in self.required_keys):
            return False
        
        # Check if all values are non-empty
        if not all(triple[key].strip() for key in self.required_keys):
            return False
        
        return True
    
    def validate_triples(
        self,
        triples: Any,
        chunk_number: int = None
    ) -> List[Dict[str, Any]]:
        """
        Validate a list of triples and add chunk information.
        
        Args:
            triples: List of potential triples to validate
            chunk_number: Optional chunk number to add to valid triples
            
        Returns:
            List of valid triples with chunk information added
        """
        if not isinstance(triples, list):
            return []
        
        valid_triples = []
        
        for triple in triples:
            if self.validate_triple(triple):
                # Add chunk information if provided
                if chunk_number is not None:
                    triple['chunk'] = chunk_number
                valid_triples.append(triple)
        
        return valid_triples
    
    def get_validation_report(
        self,
        triples: List[Any]
    ) -> Dict[str, Any]:
        """
        Generate a validation report for a list of triples.
        
        Args:
            triples: List of potential triples
            
        Returns:
            Dictionary with validation statistics
        """
        if not isinstance(triples, list):
            return {
                'total_items': 0,
                'valid_triples': 0,
                'invalid_triples': 0,
                'error': 'Input is not a list'
            }
        
        total = len(triples)
        valid = sum(1 for triple in triples if self.validate_triple(triple))
        
        return {
            'total_items': total,
            'valid_triples': valid,
            'invalid_triples': total - valid,
            'validation_rate': (valid / total * 100) if total > 0 else 0,
        }