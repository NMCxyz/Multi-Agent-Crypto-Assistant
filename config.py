"""
Configuration file for the Multi-Agent Crypto/Financial Assistant

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class AgentDecisionConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.1  # Deterministic
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.7  # Creative but factual
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.context_limit = 8     # include last 8 messages (4 Q&A pairs) in history - optimized for web search

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "crypto_financial_database"  # Collection from semantic chunking
        # Note: chunk_size and chunk_overlap are handled by semantic chunking in the ingestion notebook
        # self.embedding_model = "text-embedding-3-large"
        # Initialize OpenAI Embeddings
        self.embedding_model = OpenAIEmbeddings(
            model = "text-embedding-3-small",  # Cost-effective embedding model
            openai_api_key = os.getenv("embedding_openai_api_key")  # Replace with your OpenAI API key
        )
        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.summarizer_model = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.5  # Slightly creative but factual
        )
        self.chunker_model = ChatOpenAI(
            model_name = "gpt-4o-mini",  # Use gpt-4o-mini for cost-effective chunking
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.0  # factual
        )
        self.response_generator_model = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.1  # Consistent and factual for response generation
        )
        self.top_k = 3
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # NOTE: Routing is now based on LLM context validation, not confidence thresholds

        self.context_limit = 12     # include last 12 messages (6 Q&A pairs) in history - optimized for RAG retrieval

class CryptoAnalysisConfig:
    def __init__(self):
        # Crypto sentiment analysis endpoint
        self.crypto_sentiment_endpoint = "https://rg-crypto-bert.eastasia.inference.ml.azure.com/score"
        self.crypto_bert_api_key = os.getenv("CRYPTO_BERT_API_KEY")

        # Forward-looking statement detection endpoint
        self.fls_azure_endpoint = "https://rg-finbert-cls.eastasia.inference.ml.azure.com/score"
        self.azure_ml_api_key = os.getenv("AZURE_ML_API_KEY")

        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name (e.g., gpt-4o)
            openai_api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.1  # Keep deterministic for classification tasks
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "CRYPTO_SENTIMENT_AGENT": False,
            "FORWARD_LOOKING_STATEMENT_AGENT": False
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = False  # Disabled - speech features removed
        self.enable_image_upload = True

class CacheConfig:
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
        self.session_memory_ttl_hours = int(os.getenv("SESSION_MEMORY_TTL_HOURS", "2"))
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"

class MongoDBConfig:
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("MONGODB_DATABASE", "crypto_financial_memory")
        self.enable_memory = bool(self.mongodb_uri)

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.crypto_analysis = CryptoAnalysisConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.cache = CacheConfig()
        self.mongodb = MongoDBConfig()
        self.max_conversation_history = 20  # Include last 20 messages (10 Q&A pairs) in history

# # Example usage
# config = Config()