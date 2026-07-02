"""
Shared pytest configuration.

Fake required env vars are set at module import time (before any test module
imports src.config) so Settings() never touches real API keys or the
developer's local .env values. pydantic-settings gives real environment
variables priority over the .env file, so these values win regardless of
what's in the project's .env.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
