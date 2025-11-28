import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptsLoader:
    """
    Singleton class to load and manage prompt templates from YAML file.
    """
    _instance = None
    _prompts = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptsLoader, cls).__new__(cls)
            cls._instance._load_prompts()
        return cls._instance

    def _load_prompts(self):
        """Load prompts from YAML file."""
        try:
            # Get the path to prompts_template.yaml
            current_dir = Path(__file__).parent.parent
            prompts_file = current_dir.parent / "config" / "prompts_template.yaml"
            
            if not prompts_file.exists():
                logger.error(f"Prompts file not found at: {prompts_file}")
                raise FileNotFoundError(f"prompts_template.yaml not found at {prompts_file}")
            
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
            
            logger.info(f"✓ Successfully loaded prompts from {prompts_file}")
            logger.info(f"✓ Available prompts: {list(self._prompts.keys())}")
            
        except Exception as e:
            logger.exception(f"✗ Failed to load prompts: {e}")
            raise

    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            prompt_name: Name of the prompt template
            
        Returns:
            Prompt template string
            
        Raises:
            KeyError: If prompt_name doesn't exist
        """
        if self._prompts is None:
            self._load_prompts()
        
        if prompt_name not in self._prompts:
            available = list(self._prompts.keys())
            raise KeyError(
                f"Prompt '{prompt_name}' not found. Available prompts: {available}"
            )
        
        return self._prompts[prompt_name]

    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get and format a prompt template with provided variables.
        
        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to format the template with
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt(prompt_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable for prompt formatting: {e}")
            raise

    def reload_prompts(self):
        """Reload prompts from file (useful for development)."""
        self._load_prompts()


# Create a global instance
prompts_loader = PromptsLoader()


# Convenience functions
def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name."""
    return prompts_loader.get_prompt(prompt_name)


def format_prompt(prompt_name: str, **kwargs) -> str:
    """Get and format a prompt template."""
    return prompts_loader.format_prompt(prompt_name, **kwargs)