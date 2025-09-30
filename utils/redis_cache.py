"""
Redis Cache Manager for optimizing API calls and session memory.

This module provides caching functionality for:
1. Prompt/Response caching to avoid redundant LLM calls
2. Session memory caching for conversation history
3. Cache quality control and cost management
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import redis

class RedisCacheManager:
    """
    Redis-based cache manager for prompts, responses, and session memory.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, db: Optional[int] = None,
                 password: Optional[str] = None, decode_responses: bool = True):
        """
        Initialize Redis cache manager.

        Args:
            host: Redis host (uses REDIS_HOST env var if not provided)
            port: Redis port (uses REDIS_PORT env var if not provided)
            db: Redis database number (uses REDIS_DB env var if not provided)
            password: Redis password (uses REDIS_PASSWORD env var if not provided)
            decode_responses: Whether to decode responses as strings
        """
        self.logger = logging.getLogger(__name__)

        # Use environment variables as defaults
        import os
        redis_host = host or os.getenv('REDIS_HOST', 'localhost')
        redis_port = port or int(os.getenv('REDIS_PORT', '6379'))
        redis_db = db or int(os.getenv('REDIS_DB', '0'))
        redis_password = password or os.getenv('REDIS_PASSWORD')

        self.client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=decode_responses,
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Test connection
        self.client.ping()
        self.logger.info("Redis cache manager initialized successfully")



    def _normalize_context_for_cache(self, context: str) -> str:
        """Normalize context for consistent cache key generation."""
        if not context:
            return ""

        # Split context into chunks (assuming chunks are separated by --- markers)
        chunks = context.split("---")
        normalized_chunks = []

        for chunk in chunks:
            if chunk.strip():
                # Extract only the main content, remove metadata and formatting
                lines = chunk.strip().split('\n')
                content_lines = []

                for line in lines:
                    line = line.strip()
                    # Skip metadata lines (start with ## but are not headers)
                    if line.startswith('##') and len(line) < 50:  # Headers are usually short
                        continue
                    # Skip empty lines and separator lines
                    if line and not line.startswith('=') and not line.startswith('-'):
                        content_lines.append(line)

                if content_lines:
                    normalized_chunks.append('\n'.join(content_lines))

        # Sort chunks by content hash for consistent ordering
        sorted_chunks = sorted(normalized_chunks, key=lambda x: hashlib.md5(x.encode()).hexdigest())

        return "---".join(sorted_chunks)

    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key using hash of input data."""
        # Normalize context for consistent caching
        if "context" in data:
            data = data.copy()
            data["context"] = self._normalize_context_for_cache(data["context"])

        # Convert data to JSON string for consistent hashing
        data_str = json.dumps(data, sort_keys=True)
        # Create hash
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        return f"{prefix}:{data_hash}"

    def get_prompt_response_cache(self, query: str, context: str,
                                 chat_history: Optional[List[Dict[str, str]]] = None,
                                 session_id: Optional[str] = None) -> Optional[str]:
        """
        Get cached response for a prompt.

        Args:
            query: User query
            context: Retrieved context
            chat_history: Conversation history

        Returns:
            Cached response if found, None otherwise
        """
        # Create cache key from query, context, recent chat history, and session_id for isolation
        cache_data = {
            "query": query,
            "context": context,
            "chat_history": chat_history[-3:] if chat_history else [],  # Last 3 messages for context
            "session_id": session_id  # Include session_id to ensure session isolation
        }

        cache_key = self._generate_cache_key("prompt_response", cache_data)

        cached_response = self.client.get(cache_key)
        if cached_response:
            return cached_response
        return None

    def set_prompt_response_cache(self, query: str, context: str, response: str,
                                 chat_history: Optional[List[Dict[str, str]]] = None,
                                 session_id: Optional[str] = None,
                                 ttl_hours: int = 24) -> bool:
        """
        Cache a prompt response pair.

        Args:
            query: User query
            context: Retrieved context
            response: Generated response
            chat_history: Conversation history
            ttl_hours: Time to live in hours

        Returns:
            True if cached successfully, False otherwise
        """
        # Create cache key with session isolation
        cache_data = {
            "query": query,
            "context": context,
            "chat_history": chat_history[-3:] if chat_history else [],
            "session_id": session_id  # Include session_id to ensure session isolation
        }

        cache_key = self._generate_cache_key("prompt_response", cache_data)

        ttl_seconds = ttl_hours * 3600
        self.client.setex(cache_key, ttl_seconds, response)
        return True

    def get_session_memory(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get cached session memory.

        Args:
            session_id: Unique session identifier

        Returns:
            Cached conversation history if found, None otherwise
        """
        cache_key = f"session_memory:{session_id}"

        cached_memory = self.client.get(cache_key)
        if cached_memory:
            memory_data = json.loads(cached_memory)
            return memory_data
        return None

    def set_session_memory(self, session_id: str, memory: List[Dict[str, str]],
                          ttl_hours: int = 1) -> bool:
        """
        Cache session memory.

        Args:
            session_id: Unique session identifier
            memory: Conversation history
            ttl_hours: Time to live in hours

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = f"session_memory:{session_id}"

        # Keep only last 15 messages to avoid memory bloat and optimize performance
        memory_data = memory[-15:] if len(memory) > 15 else memory
        memory_json = json.dumps(memory_data)

        ttl_seconds = ttl_hours * 3600
        self.client.setex(cache_key, ttl_seconds, memory_json)
        return True

    def update_session_memory(self, session_id: str, user_message: str,
                            assistant_message: str) -> bool:
        """
        Update session memory with new conversation turn.

        Args:
            session_id: Unique session identifier
            user_message: Latest user message
            assistant_message: Latest assistant response

        Returns:
            True if updated successfully, False otherwise
        """
        # Get existing memory
        existing_memory = self.get_session_memory(session_id) or []

        # Add new turn
        new_turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        updated_memory = existing_memory + new_turn

        # Cache updated memory
        return self.set_session_memory(session_id, updated_memory)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        info = self.client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "evicted_keys": info.get("evicted_keys", 0)
        }

    def check_cache_quality(self, query: str, context: str, cached_response: str,
                           confidence_threshold: float = 0.7) -> bool:
        """
        Check if cached response is of sufficient quality to avoid LLM calls.

        Args:
            query: Original user query
            context: Retrieved context
            cached_response: Cached response to evaluate
            confidence_threshold: Minimum confidence threshold

        Returns:
            True if cache quality is sufficient, False otherwise
        """
        # Simple heuristic checks
        # 1. Response should not be too short (likely error response)
        if len(cached_response.strip()) < 50:
            return False

        # 2. Response should not contain error indicators
        error_indicators = [
            "error", "failed", "unable to", "cannot", "not available",
            "insufficient", "no information", "don't have enough"
        ]
        response_lower = cached_response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                return False

        # 3. Response should be relevant to query (basic keyword check)
        query_keywords = set(query.lower().split()[:5])  # First 5 words
        response_keywords = set(cached_response.lower().split()[:20])  # First 20 words of response

        # At least 1 keyword overlap
        if not query_keywords.intersection(response_keywords):
            return False

        # 4. Context relevance check (if context is provided)
        if context and len(context.strip()) > 100:
            context_keywords = set(context.lower().split()[:10])
            if not query_keywords.intersection(context_keywords):
                return False

        return True

    def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """
        Clear cache entries.

        Args:
            pattern: Pattern to match keys (e.g., "prompt_response:*")

        Returns:
            True if cleared successfully, False otherwise
        """
        if pattern:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        else:
            self.client.flushdb()

        return True

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> RedisCacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
    return _cache_manager

def reset_cache_manager():
    """Reset global cache manager (for testing)."""
    global _cache_manager
    _cache_manager = None
