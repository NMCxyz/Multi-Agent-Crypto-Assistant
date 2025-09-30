"""
Memory Management Tools for the Multi-Agent Crypto/Financial Assistant

This module provides tools for agents to manage their long-term memory:
1. Save important information to memory
2. Retrieve relevant memories
3. Update existing memories
4. Delete outdated memories
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_core.tools import tool
from utils.mongodb_manager import get_mongodb_manager

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory management utilities for agents.
    """

    def __init__(self):
        """Initialize memory manager."""
        self.mongodb_manager = get_mongodb_manager()
        self.store = self.mongodb_manager.get_store()

    def save_memory(self, content: str, memory_type: str = "general",
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save information to long-term memory.

        Args:
            content: The content to remember
            memory_type: Type of memory (user_preferences, conversation_summary, etc.)
            metadata: Additional metadata

        Returns:
            True if saved successfully, False otherwise
        """
        # Create memory data
        memory_data = {
            "content": content,
            "memory_type": memory_type,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Save to MongoDB store
        self.store.put(
            namespace=("memories", memory_type),
            key=f"memory_{datetime.utcnow().timestamp()}",
            value=memory_data
        )

        logger.info(f"Saved memory: {memory_type} - {content[:50]}...")
        return True

    def get_relevant_memories(self, query: str, memory_type: str = "general",
                           limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant memories based on a query.

        Args:
            query: Search query
            memory_type: Type of memory to search
            limit: Maximum number of results

        Returns:
            List of relevant memories
        """
        # Search for relevant memories
        results = self.store.search(
            ("memories", memory_type),  # namespace_prefix as positional argument
            query=query,
            limit=limit
        )

        # Convert to simple dict format
        memories = []
        for result in results:
            memory_data = {
                "content": result.content,
                "memory_type": result.metadata.get("memory_type", memory_type),
                "created_at": result.metadata.get("created_at", ""),
                "score": getattr(result, 'score', 0.0)
            }
            memories.append(memory_data)

        return memories

    def update_memory(self, memory_key: str, new_content: str,
                     memory_type: str = "general") -> bool:
        """
        Update existing memory.

        Args:
            memory_key: Key of memory to update
            new_content: New content
            memory_type: Type of memory

        Returns:
            True if updated successfully, False otherwise
        """
        # Get existing memory
        existing = self.store.get(namespace=("memories", memory_type), key=memory_key)

        if existing:
            # Update content and timestamp
            updated_data = existing.value.copy()
            updated_data["content"] = new_content
            updated_data["updated_at"] = datetime.utcnow().isoformat()

            # Save updated memory
            self.store.put(
                namespace=("memories", memory_type),
                key=memory_key,
                value=updated_data
            )

            logger.info(f"Updated memory: {memory_key}")
            return True

        return False

    def delete_memory(self, memory_key: str, memory_type: str = "general") -> bool:
        """
        Delete a memory.

        Args:
            memory_key: Key of memory to delete
            memory_type: Type of memory

        Returns:
            True if deleted successfully, False otherwise
        """
        # Delete memory
        self.store.delete(namespace=("memories", memory_type), key=memory_key)
        logger.info(f"Deleted memory: {memory_key}")
        return True

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def reset_memory_manager():
    """Reset global memory manager (for testing)."""
    global _memory_manager
    _memory_manager = None

# Create memory management tools for agents
@tool
def save_user_preference(preference: str, description: str = "") -> str:
    """
    Save user preference to long-term memory.

    Args:
        preference: The user preference to remember
        description: Additional description of the preference

    Returns:
        Confirmation message
    """
    memory_manager = get_memory_manager()
    content = f"User preference: {preference}"
    if description:
        content += f" - {description}"

    memory_manager.save_memory(
        content=content,
        memory_type="user_preferences",
        metadata={"preference_type": preference}
    )

    return "User preference saved to memory"

@tool
def get_user_preferences(limit: int = 3) -> str:
    """
    Get user's saved preferences from memory.

    Args:
        limit: Maximum number of preferences to return

    Returns:
        String containing user's preferences
    """
    memory_manager = get_memory_manager()
    memories = memory_manager.get_relevant_memories(
        query="user preferences",
        memory_type="user_preferences",
        limit=limit
    )

    result = "User preferences:\n"
    for memory in memories:
        result += f"- {memory['content']}\n"

    return result

@tool
def save_conversation_summary(summary: str, topics: List[str] = None) -> str:
    """
    Save conversation summary to memory for future reference.

    Args:
        summary: Summary of the conversation
        topics: Key topics discussed

    Returns:
        Confirmation message
    """
    memory_manager = get_memory_manager()

    metadata = {}
    if topics:
        metadata["topics"] = topics

    memory_manager.save_memory(
        content=f"Conversation summary: {summary}",
        memory_type="conversation_summaries",
        metadata=metadata
    )

    return "Conversation summary saved to memory"

@tool
def get_conversation_history(limit: int = 3) -> str:
    """
    Get recent conversation summaries from memory.

    Args:
        limit: Maximum number of summaries to return

    Returns:
        String containing conversation history
    """
    memory_manager = get_memory_manager()
    memories = memory_manager.get_relevant_memories(
        query="conversation summary",
        memory_type="conversation_summaries",
        limit=limit
    )

    result = "Recent conversation summaries:\n"
    for memory in memories:
        result += f"- {memory['content']}\n"

    return result
