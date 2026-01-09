"""
AgentLLM - Interface for Ollama LLM models.
Lists and loads Ollama models for RAG applications.
"""
import requests
from typing import List, Optional
import json


class AgentLLM:
    """Interface for interacting with Ollama LLM models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: Optional[str] = None):
        """
        Initialize the Ollama LLM interface.
        
        Args:
            base_url: Base URL for Ollama API
            model_name: Name of the model to use (optional, can be set later)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify that Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {str(e)}"
            )
    
    def list_models(self) -> List[str]:
        """
        List all available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            if 'models' in data:
                for model in data['models']:
                    model_name = model.get('name', '')
                    # Remove ':latest' suffix if present for cleaner display
                    if ':' in model_name:
                        model_name = model_name.split(':')[0]
                    models.append(model_name)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_models = []
            for model in models:
                if model not in seen:
                    seen.add(model)
                    unique_models.append(model)
            
            return sorted(unique_models)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list Ollama models: {str(e)}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        if self.model_name is None:
            raise ValueError("No model selected. Use set_model() or provide model_name in constructor.")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'message' in data and 'content' in data['message']:
                return data['message']['content']
            else:
                raise RuntimeError(f"Unexpected response format: {data}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def set_model(self, model_name: str):
        """
        Set the model to use.
        
        Args:
            model_name: Name of the Ollama model
        """
        available_models = self.list_models()
        
        # Check if model exists (with or without :latest suffix)
        model_found = False
        for available in available_models:
            if available == model_name or f"{model_name}:latest" in available:
                model_found = True
                break
        
        if not model_found:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
            )
        
        self.model_name = model_name
    
    def get_model(self) -> Optional[str]:
        """Get the currently selected model name."""
        return self.model_name

