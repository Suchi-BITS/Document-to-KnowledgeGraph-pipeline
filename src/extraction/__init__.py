"""
Triple extraction module.
Extracts and validates SPO triples from text.
"""

from .extractor import TripleExtractor
from .validator import TripleValidator

__all__ = ['TripleExtractor', 'TripleValidator']
