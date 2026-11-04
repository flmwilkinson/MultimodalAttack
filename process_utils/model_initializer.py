#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoupled Model Initializer
===========================

Model initializer based on the project root config (config.py), replacing RIVAL/modules/model_initializer.
Infers the model type from MODEL_REGISTRY and creates clients using API_KEYS / API_URLS.
"""

from typing import Dict
from openai import OpenAI

# Uniformly import the model registry from the project root config
from config import MODEL_REGISTRY


# Cloud providers whose registry names carry a "provider/" prefix used only for
# type detection -- the prefix must be stripped before the API call. Local
# Ollama tags that legitimately contain a slash (e.g. "huihui_ai/...:q8_0",
# "x/z-image-turbo:latest") are NOT touched, because they don't start with one
# of these provider prefixes.
_CLOUD_PREFIXES = ("openai/", "openrouter/", "siliconcloud/", "anthropic/")


def api_model_name(model_name: str) -> str:
    """Return the model id to send to the provider's API.

    For cloud providers the registry name is e.g. "openrouter/qwen/qwen3-vl-..."
    but the API expects "qwen/qwen3-vl-..."; strip the leading provider segment.
    Local/Ollama names are returned unchanged.
    """
    if not model_name:
        return model_name
    for p in _CLOUD_PREFIXES:
        if model_name.startswith(p):
            return model_name[len(p):]
    return model_name


class ModelNotSupportedError(Exception):
    """The requested model type is not supported"""
    pass


class ModelInitializer:
    # Use MODEL_REGISTRY from the project root config
    MODEL_REGISTRY = MODEL_REGISTRY

    def __init__(self, model_config: Dict, initialize_client: bool = True):
        """
        Initialize the model

        Parameters:
            model_config: dictionary containing the following keys:
                - model_name: Model name (e.g., openai/gpt-4o-mini)
                - api_key: API key (optional; if not provided, read from the environment or unified config by type)
                - api_url: API base URL (optional)
                - temperature: Sampling temperature (optional)
                - max_tokens: Maximum generation length (optional)
            initialize_client: Whether to initialize the client; defaults to True
        """
        if "model_name" not in model_config:
            raise ValueError("model_name is required in config")

        self.model_name = model_config["model_name"]
        self.model_type = self._determine_model_type(self.model_name)

        self.client = None
        if initialize_client:
            self.client = self._initialize_client(model_config)

    def _determine_model_type(self, model_name: str) -> str:
        """Infer the model type from MODEL_REGISTRY"""
        for model_type, models in self.MODEL_REGISTRY.items():
            if model_name in models:
                return model_type
        raise ModelNotSupportedError(f"Unsupported model: {model_name}")

    def _initialize_client(self, config: Dict):
        """Create the model client"""
        try:
            if self.model_type in ("openai", "ollama"):
                if "api_key" not in config:
                    raise ValueError("OpenAI API key is required")
                client = OpenAI(
                    api_key=config.get("api_key"),
                    base_url=config.get("api_url")
                )
                return client

            elif self.model_type == "anthropic":
                # Avoid top-level import; load on demand
                try:
                    import anthropic
                except Exception as e:
                    raise ImportError(f"Anthropic client not available: {e}")
                if "api_key" not in config:
                    raise ValueError("Anthropic API key is required")
                client = anthropic.Client(
                    api_key=config.get("api_key"),
                    base_url=config.get("api_url")
                )
                return client

            elif self.model_type == "siliconcloud":
                if "api_key" not in config:
                    raise ValueError("SiliconCloud API key is required")
                client = OpenAI(
                    api_key=config.get("api_key"),
                    base_url=config.get("api_url")
                )
                return client

            elif self.model_type == "openrouter":
                if "api_key" not in config:
                    raise ValueError("OpenRouter API key is required")
                client = OpenAI(
                    api_key=config.get("api_key"),
                    base_url=config.get("api_url"),
                )
                return client

            elif self.model_type in ("commercial", "local"):
                raise NotImplementedError(f"{self.model_type} API support not implemented yet")
            else:
                raise ModelNotSupportedError(f"Model type {self.model_type} initialization not implemented")

        except Exception as e:
            raise Exception(f"Failed to create model client: {str(e)}")

    def get_roles(self) -> tuple:
        """Return role names (user, assistant)"""
        if self.model_type == "commercial":
            if self.model_name == "kuaishou":
                return ("user", "kuaiyi")
            elif self.model_name in ["zhipu", "douyin", "wenxinyiyan", "baichuan"]:
                return ("user", "assistant")
        return ("user", "assistant")

    def get_message_format(self, message: str, role: str = "user") -> Dict:
        """Unified message formatting"""
        if not isinstance(message, str):
            raise ValueError("message must be a string")

        roles = self.get_roles()
        if role == "assistant":
            role = roles[1]
        elif role == "user":
            role = roles[0]
        elif role != "system":
            raise ValueError(f"Invalid role: {role}")

        return {"role": role, "content": message}

    def validate_config(self, config: Dict) -> None:
        """Validate configuration parameters"""
        required_params = {
            "openai": ["api_key"],
            "ollama": ["api_key"],
            "anthropic": ["api_key"],
            "siliconcloud": ["api_key", "api_url"],
            "commercial": [],
            "local": [],
            "openrouter": ["api_key", "api_url"],
        }

        if self.model_type in required_params:
            missing = [param for param in required_params[self.model_type]
                       if param not in config]
            if missing:
                raise ValueError(f"Missing required parameters for {self.model_type}: {missing}")

    @classmethod
    def get_supported_models(cls) -> Dict:
        """Return the list of supported models"""
        return cls.MODEL_REGISTRY.copy()