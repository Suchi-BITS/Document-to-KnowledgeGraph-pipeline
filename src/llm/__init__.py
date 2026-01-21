"""
LLM interaction module.
Handles communication with Large Language Models.
"""

from .client import LLMClient
from .prompts import PromptTemplates

__all__ = ['LLMClient', 'PromptTemplates']
