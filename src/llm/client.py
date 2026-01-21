"""
LLM Client for interacting with OpenAI-compatible APIs.
Handles API calls, error handling, and response processing.
"""

import openai
from typing import Optional, Dict, Any
from config.settings import settings


class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for authentication (defaults to settings)
            base_url: Base URL for the API (defaults to settings)
            model: Model name to use (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Maximum tokens in response (defaults to settings)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_API_BASE
        self.model = model or settings.LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request to the LLM.
        
        Args:
            system_prompt: System message setting the context/role
            user_prompt: User message with the actual request
            response_format: Optional response format specification
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the full API response
            
        Raises:
            Exception: If the API call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format
        
        # Add any additional parameters
        request_params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(**request_params)
            return response
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def extract_content(self, response: Any) -> str:
        """
        Extract the text content from an API response.
        
        Args:
            response: The API response object
            
        Returns:
            The extracted content as a string
        """
        try:
            return response.choices[0].message.content.strip()
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Failed to extract content from response: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }