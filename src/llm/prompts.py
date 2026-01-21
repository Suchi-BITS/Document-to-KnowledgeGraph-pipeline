"""
Prompt templates for LLM-based knowledge graph extraction.
Contains system and user prompts for extracting SPO triples.
"""


class PromptTemplates:
    """Collection of prompt templates for knowledge graph extraction."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """
        Get the system prompt that sets the LLM's role and context.
        
        Returns:
            System prompt string
        """
        return """
You are an AI expert specialized in knowledge graph extraction. 
Your task is to identify and extract factual Subject-Predicate-Object (SPO) triples from the given text.
Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt.
Extract core entities and the most direct relationship.
"""
    
    @staticmethod
    def get_user_prompt_template() -> str:
        """
        Get the user prompt template for extraction.
        
        Returns:
            User prompt template string with {text_chunk} placeholder
        """
        return """
Please extract Subject-Predicate-Object (S-P-O) triples from the text below.

**VERY IMPORTANT RULES:**
1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
7.  **Completeness:** Extract all distinct factual relationships mentioned.

**Text to Process:**
```text
{text_chunk}
```

**Required JSON Output Format Example:**
[
  {{ "subject": "marie curie", "predicate": "discovered", "object": "radium" }},
  {{ "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }}
]

**Your JSON Output (MUST start with '[' and end with ']'):**
"""
    
    @staticmethod
    def format_user_prompt(text_chunk: str) -> str:
        """
        Format the user prompt with actual text chunk.
        
        Args:
            text_chunk: The text to extract triples from
            
        Returns:
            Formatted user prompt
        """
        template = PromptTemplates.get_user_prompt_template()
        return template.format(text_chunk=text_chunk)
    
    @staticmethod
    def get_prompts_for_chunk(text_chunk: str) -> tuple[str, str]:
        """
        Get both system and formatted user prompts for a text chunk.
        
        Args:
            text_chunk: The text to extract triples from
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = PromptTemplates.get_system_prompt()
        user_prompt = PromptTemplates.format_user_prompt(text_chunk)
        return system_prompt, user_prompt