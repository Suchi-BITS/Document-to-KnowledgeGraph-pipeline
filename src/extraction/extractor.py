"""
Triple extraction module for extracting SPO triples from text chunks using LLMs.
Handles API calls, JSON parsing, and error recovery.
"""

import json
import re
from typing import List, Dict, Optional
from src.llm.client import LLMClient
from src.llm.prompts import PromptTemplates
from src.extraction.validator import TripleValidator


class TripleExtractor:
    """Extracts Subject-Predicate-Object triples from text using LLMs."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the triple extractor.
        
        Args:
            llm_client: Initialized LLM client for API calls
        """
        self.llm_client = llm_client
        self.validator = TripleValidator()
        self.failed_chunks = []
    
    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_number: int
    ) -> Dict[str, any]:
        """
        Extract triples from a single text chunk.
        
        Args:
            chunk_text: The text to extract triples from
            chunk_number: Sequential number of this chunk
            
        Returns:
            Dictionary containing:
            - triples: List of valid extracted triples
            - raw_response: Raw LLM output
            - parsed_json: Parsed JSON data
            - error: Error message if extraction failed
        """
        result = {
            'chunk_number': chunk_number,
            'triples': [],
            'raw_response': None,
            'parsed_json': None,
            'error': None,
        }
        
        try:
            # Get prompts
            system_prompt, user_prompt = PromptTemplates.get_prompts_for_chunk(chunk_text)
            
            # Make API call
            response = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"}
            )
            
            # Extract raw content
            raw_output = self.llm_client.extract_content(response)
            result['raw_response'] = raw_output
            
            # Parse JSON
            parsed_json = self._parse_json_response(raw_output)
            result['parsed_json'] = parsed_json
            
            if parsed_json is None:
                result['error'] = 'JSON parsing failed'
                self.failed_chunks.append(result)
                return result
            
            # Validate and extract triples
            valid_triples = self.validator.validate_triples(parsed_json, chunk_number)
            result['triples'] = valid_triples
            
        except Exception as e:
            result['error'] = f'Extraction error: {str(e)}'
            self.failed_chunks.append(result)
        
        return result
    
    def extract_from_chunks(
        self,
        chunks: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Extract triples from multiple text chunks.
        
        Args:
            chunks: List of chunk dictionaries from TextChunker
            
        Returns:
            List of all valid extracted triples
        """
        all_triples = []
        self.failed_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_number = chunk['chunk_number']
            
            result = self.extract_from_chunk(chunk_text, chunk_number)
            
            if result['triples']:
                all_triples.extend(result['triples'])
        
        return all_triples
    
    def _parse_json_response(self, raw_output: str) -> Optional[List[Dict]]:
        """
        Parse JSON from LLM response with fallback strategies.
        
        Args:
            raw_output: Raw string output from LLM
            
        Returns:
            Parsed list of dictionaries or None if parsing fails
        """
        try:
            # Strategy 1: Direct parsing
            parsed_data = json.loads(raw_output)
            
            # Handle response_format={'type':'json_object'} that returns a dict
            if isinstance(parsed_data, dict):
                # Extract list from dictionary values
                list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    return list_values[0]
                else:
                    return None
            elif isinstance(parsed_data, list):
                return parsed_data
            else:
                return None
                
        except json.JSONDecodeError:
            # Strategy 2: Regex fallback for arrays wrapped in text/markdown
            match = re.search(r'^\s*(\[.*?\])\s*$', raw_output, re.DOTALL)
            if match:
                json_string_extracted = match.group(1)
                try:
                    return json.loads(json_string_extracted)
                except json.JSONDecodeError:
                    return None
            return None
    
    def get_failed_chunks(self) -> List[Dict[str, any]]:
        """
        Get information about chunks that failed extraction.
        
        Returns:
            List of failed chunk information
        """
        return self.failed_chunks
    
    def get_extraction_statistics(
        self,
        total_chunks: int,
        total_triples: int
    ) -> Dict[str, any]:
        """
        Get extraction statistics.
        
        Args:
            total_chunks: Total number of chunks processed
            total_triples: Total number of triples extracted
            
        Returns:
            Dictionary with statistics
        """
        return {
            'total_chunks': total_chunks,
            'failed_chunks': len(self.failed_chunks),
            'successful_chunks': total_chunks - len(self.failed_chunks),
            'total_triples_extracted': total_triples,
            'avg_triples_per_chunk': total_triples / total_chunks if total_chunks > 0 else 0,
        }