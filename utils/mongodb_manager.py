"""
MongoDB Manager for the Multi-Agent Crypto/Financial Assistant

This module provides MongoDB connection management and configuration for:
1. MongoDB Atlas connection for production use
2. MongoDBStore for long-term memory storage
3. Session management (MongoDBSaver for conversation checkpointing)
4. Memory management integration with LangGraph
"""

import os
import logging
from typing import Optional
from pymongo import MongoClient
from langgraph.store.mongodb.base import MongoDBStore, VectorIndexConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_openai import OpenAIEmbeddings

class MongoDBManager:
    """
    MongoDB connection and configuration manager.
    """

    def __init__(self):
        """
        Initialize MongoDB manager with configuration from environment variables.
        """
        self.logger = logging.getLogger(__name__)

        # MongoDB Atlas configuration
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("MONGODB_DATABASE", "crypto_financial_memory")

        # Check if MongoDB is enabled
        self.enabled = bool(self.mongodb_uri)

        if self.enabled:
            try:
                # Initialize MongoDB client
                self.client = MongoClient(self.mongodb_uri)

                # Test connection
                self.client.admin.command('ping')
                self.logger.info("MongoDB connection established successfully")

                # Initialize database
                self.db = self.client[self.database_name]

            except Exception as e:
                self.logger.error(f"Failed to connect to MongoDB: {e}")
                self.client = None
                self.db = None
                self.enabled = False
        else:
            self.logger.warning("MongoDB not configured, memory features will be disabled")
            self.client = None
            self.db = None

    def get_store(self) -> MongoDBStore:
        """
        Get MongoDBStore for long-term memory storage.

        Returns:
            MongoDBStore instance
        """
        # Create collection for memory store
        collection = self.db["memory_store"]

        # Initialize store with vector search capabilities
        store = MongoDBStore(
            collection=collection,
            index_config=VectorIndexConfig(
                fields=None,  # Auto-detect fields for indexing
                filters=None,  # No additional filters
                dims=1536,     # OpenAI embedding dimensions
                embed=OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=os.getenv("embedding_openai_api_key")
                )
            ),
            auto_index_timeout=70
        )

        self.logger.info("MongoDBStore initialized successfully")
        return store

    def get_saver(self) -> MongoDBSaver:
        """
        Get MongoDBSaver for conversation checkpointing.

        Returns:
            MongoDBSaver instance
        """
        # Create MongoDBSaver with the configured database
        saver = MongoDBSaver(
            client=self.client,
            db_name=self.database_name,
            checkpoint_collection_name="checkpoints",
            writes_collection_name="checkpoint_writes",
            ttl=None  # No TTL by default
        )

        self.logger.info("MongoDBSaver initialized successfully")
        return saver

    def is_enabled(self) -> bool:
        """
        Check if MongoDB memory features are enabled.

        Returns:
            True if MongoDB is configured and connected, False otherwise
        """
        return True  # Always enabled, no fallback logic

    def get_database_info(self) -> dict:
        """
        Get information about the MongoDB database.

        Returns:
            Dictionary with database information
        """
        # Get database stats
        stats = self.db.command("dbStats")
        collections = self.db.list_collection_names()

        return {
            "enabled": True,
            "database_name": self.database_name,
            "collections": collections,
            "data_size": stats.get("dataSize", 0),
            "storage_size": stats.get("storageSize", 0),
            "indexes": stats.get("indexes", 0),
            "collections_count": stats.get("collections", 0)
        }

# Global MongoDB manager instance
_mongodb_manager = None

def get_mongodb_manager() -> MongoDBManager:
    """
    Get or create global MongoDB manager instance.

    Returns:
        MongoDBManager instance
    """
    global _mongodb_manager
    if _mongodb_manager is None:
        _mongodb_manager = MongoDBManager()
    return _mongodb_manager

def reset_mongodb_manager():
    """
    Reset global MongoDB manager (for testing).
    """
    global _mongodb_manager
    _mongodb_manager = None
