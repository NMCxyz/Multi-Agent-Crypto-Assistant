"""
🧠 Semantic Chunking Data Ingestion Script for Qdrant Cloud

This script provides advanced document ingestion with:
- 🧠 LLM-based semantic chunking for intelligent document splitting
- 📊 Detailed progress tracking and statistics
- 🔄 Hierarchical processing for large documents (200+ pages)
- ⚡ Error resilience with fallback mechanisms
- 🔗 Direct Qdrant Cloud or local database support

Usage:
    python ingest_rag_data.py --file <file_path>        # Ingest single file
    python ingest_rag_data.py --dir <directory_path>    # Ingest all files in directory
    python ingest_rag_data.py --reset                   # Reset collection before ingestion
"""

import sys
import json
import logging
import time
import os
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from agents.rag_agent import CryptoFinancialRAG
from agents.rag_agent.doc_parser import CryptoFinancialDocParser
from agents.rag_agent.content_processor import ContentProcessor
from config import Config

# Qdrant imports
from qdrant_client import QdrantClient, models
from langchain_qdrant import FastEmbedSparse

import argparse

# Initialize parser
parser = argparse.ArgumentParser(
    description="🧠 Semantic Chunking Data Ingestion for Qdrant",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")
parser.add_argument("--reset", action="store_true", help="Reset collection before ingestion")
parser.add_argument("--analyze", action="store_true", help="Analyze collection after ingestion")

# Parse arguments
args = parser.parse_args()

# Load configuration
config = Config()

# Display configuration
print("\n" + "="*70)
print("🔧 CONFIGURATION")
print("="*70)

qdrant_url = config.rag.url
qdrant_api_key = config.rag.api_key
collection_name = config.rag.collection_name

print(f"📊 Collection Name: {collection_name}")
print(f"🔍 Qdrant URL: {qdrant_url if qdrant_url else 'localhost'}")
print(f"🔑 API Key: {'***' + qdrant_api_key[-4:] if qdrant_api_key else 'Not configured (using local)'}")

if qdrant_url and qdrant_api_key:
    print(f"☁️  Mode: Qdrant Cloud")
else:
    print(f"💾 Mode: Local Qdrant Database")

print("="*70 + "\n")


def estimate_tokens(text: str) -> int:
    """Estimate token count using the standard 4 characters = 1 token approximation."""
    return len(text) // 4


def create_qdrant_client(config) -> QdrantClient:
    """Create Qdrant client based on configuration."""
    if config.rag.url and config.rag.api_key:
        logger.info("Connecting to Qdrant Cloud...")
        client = QdrantClient(
            url=config.rag.url,
            api_key=config.rag.api_key
        )
    else:
        logger.info("Connecting to local Qdrant...")
        client = QdrantClient(path=config.rag.vector_local_path)

    # Test connection
    collections = client.get_collections()
    logger.info(f"✅ Connected successfully: {len(collections.collections)} collections found")
    return client


def reset_collection(client: QdrantClient, config):
    """Reset (delete and recreate) the collection."""
    collection_name = config.rag.collection_name
    vector_size = config.rag.embedding_dim

    print("\n" + "="*70)
    print("🔄 RESETTING COLLECTION")
    print("="*70)

    # Step 1: Check and delete existing collection
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if collection_name in collection_names:
        print(f"🗑️  Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        print(f"✅ Successfully deleted old collection")
    else:
        print(f"📝 No existing collection found")

    # Step 2: Create new collection with hybrid support
    print(f"🏗️  Creating new collection with hybrid support...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"✅ Collection '{collection_name}' created successfully")

    # Step 3: Verify configuration
    verify_info = client.get_collection(collection_name)
    print(f"\n📊 Collection Verification:")
    print(f"   - Name: {collection_name}")

    # Handle different vector config structures safely
    if hasattr(verify_info.config.params, 'vectors'):
        if isinstance(verify_info.config.params.vectors, dict):
            vector_keys = list(verify_info.config.params.vectors.keys())
            if vector_keys:
                vector_config = verify_info.config.params.vectors[vector_keys[0]]
                print(f"   - Vector size: {vector_config.size}")
                print(f"   - Distance: {vector_config.distance}")
        else:
            print(f"   - Vector size: {verify_info.config.params.vectors.size}")
            print(f"   - Distance: {verify_info.config.params.vectors.distance}")

    # Verify BOTH dense and sparse vectors
    sparse_check = verify_info.config.params.sparse_vectors
    if sparse_check and "sparse" in sparse_check:
        print(f"   - Sparse vectors: ✅ {list(sparse_check.keys())}")

    print(f"\n🎉 Collection ready for HYBRID RETRIEVAL!")
    print(f"   🔬 Dense vectors: For semantic similarity")
    print(f"   🔍 Sparse vectors: For keyword matching")
    print(f"   ⚡ Hybrid mode: Combines both for optimal retrieval!")
    print("="*70 + "\n")


def process_and_ingest_document(
    file_path: str, 
    client: QdrantClient,
    config,
    doc_parser: CryptoFinancialDocParser,
    content_processor: ContentProcessor
) -> Dict[str, Any]:
    """Process a document with semantic chunking and ingest to Qdrant."""
    
    print(f"\n{'='*70}")
    print(f"🚀 PROCESSING: {os.path.basename(file_path)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Phase 1: Document Parsing
    print(f"\n📋 PHASE 1: Document Parsing & Extraction")
    print(f"-" * 50)

    parsed_document, images = doc_parser.parse_document(
        file_path,
        config.rag.parsed_content_dir
    )
    print(f"✅ Parsed document and extracted {len(images)} images")

    # Phase 2: Image Summarization
    print(f"\n📋 PHASE 2: Image Summarization")
    print(f"-" * 50)

    image_summaries = content_processor.summarize_images(images)
    print(f"✅ Generated {len(image_summaries)} image summaries")

    # Phase 3: Document Formatting
    print(f"\n📋 PHASE 3: Document Formatting")
    print(f"-" * 50)

    formatted_document = content_processor.format_document_with_images(
        parsed_document,
        image_summaries
    )
    print(f"✅ Formatted document with image summaries")

    # Phase 4: Semantic Chunking
    print(f"\n📋 PHASE 4: Semantic Chunking")
    print(f"-" * 50)

    print(f"📄 Document length: {len(formatted_document)} characters")
    print(f"📊 Estimated tokens: {estimate_tokens(formatted_document)} tokens")

    document_chunks = content_processor.chunk_document(formatted_document)

    print(f"✅ Created {len(document_chunks)} semantic chunks")

    # Analyze chunk quality
    token_counts = [estimate_tokens(chunk) for chunk in document_chunks]
    print(f"\n📊 Chunk Statistics:")
    print(f"   - Total chunks: {len(document_chunks)}")
    print(f"   - Average size: {sum(token_counts) // len(token_counts) if token_counts else 0} tokens")
    print(f"   - Min size: {min(token_counts) if token_counts else 0} tokens")
    print(f"   - Max size: {max(token_counts) if token_counts else 0} tokens")

    # Phase 5: Embedding Generation
    print(f"\n📋 PHASE 5: Embedding Generation")
    print(f"-" * 50)

    embedding_model = config.rag.embedding_model
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    points = []

    for i, chunk in enumerate(tqdm(document_chunks, desc="🔮 Generating embeddings")):
        # Generate dense embedding
        dense_vector = embedding_model.embed_query(chunk)

        # Generate sparse embedding
        sparse_vector = sparse_embeddings.embed_query(chunk)

        # Create point ID
        point_id = str(uuid.uuid4())

        # Create point with hybrid vectors
        point = models.PointStruct(
            id=point_id,
            vector={
                "": dense_vector,  # Dense vector (unnamed)
                "sparse": sparse_vector  # Sparse vector
            },
            payload={
                "text": chunk,
                "source": os.path.basename(file_path),
                "source_path": file_path,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "token_count": estimate_tokens(chunk),
                "chunking_method": "semantic_llm"
            }
        )
        points.append(point)

    print(f"✅ Generated {len(points)} embeddings")

    # Phase 6: Upload to Qdrant
    print(f"\n📋 PHASE 6: Upload to Qdrant")
    print(f"-" * 50)

    collection_name = config.rag.collection_name

    with tqdm(total=1, desc="☁️  Uploading to Qdrant") as pbar:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        pbar.update(1)

    print(f"✅ Uploaded {len(points)} points to collection '{collection_name}'")

    # Final Summary
    total_time = time.time() - start_time

    print(f"\n🎉 INGESTION COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 FINAL STATISTICS:")
    print(f"   📄 Document: {os.path.basename(file_path)}")
    print(f"   🧩 Chunks created: {len(document_chunks)}")
    print(f"   🔮 Embeddings generated: {len(points)}")
    print(f"   ☁️  Points uploaded: {len(points)}")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds")
    print(f"   🚀 Avg time per chunk: {total_time/len(document_chunks):.2f} seconds")
    print(f"{'='*70}")

    return {
        "success": True,
        "file": file_path,
        "chunks": len(document_chunks),
        "points": len(points),
        "time": total_time
    }


def analyze_collection(client: QdrantClient, config):
    """Analyze the collection and show statistics."""
    collection_name = config.rag.collection_name

    print("\n" + "="*70)
    print("📊 COLLECTION ANALYSIS")
    print("="*70)

    # Get collection info
    collection_info = client.get_collection(collection_name)

    print(f"\n📈 Collection Statistics:")
    print(f"   ☁️  Collection: {collection_name}")
    print(f"   📄 Points count: {collection_info.points_count}")
    print(f"   🔮 Vectors count: {collection_info.vectors_count}")

    # Sample analysis
    sample_size = min(20, collection_info.points_count)
    if sample_size > 0:
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            with_payload=True
        )[0]

        if sample_points:
            token_counts = [point.payload.get('token_count', 0) for point in sample_points]
            avg_tokens = sum(token_counts) / len(token_counts)

            print(f"\n📊 Chunk Analysis (sample of {len(sample_points)}):")
            print(f"   - Average tokens per chunk: {avg_tokens:.1f}")
            print(f"   - Token count range: {min(token_counts)} - {max(token_counts)}")

            # Show chunking methods
            methods = {}
            sources = set()
            for point in sample_points:
                method = point.payload.get('chunking_method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
                sources.add(point.payload.get('source', 'unknown'))

            print(f"\n🔧 Chunking Methods:")
            for method, count in methods.items():
                print(f"      - {method}: {count} chunks")

            print(f"\n📚 Sources in collection:")
            for source in sources:
                print(f"      - {source}")

    print("="*70 + "\n")


def main():
    """Main ingestion function."""

    # Initialize clients and processors
    client = create_qdrant_client(config)
    doc_parser = CryptoFinancialDocParser()
    content_processor = ContentProcessor(config)

    # Reset collection if requested
    if args.reset:
        reset_collection(client, config)

    # Collect files to process
    files_to_process = []

    if args.file:
        if os.path.exists(args.file):
            files_to_process.append(args.file)
        else:
            logger.error(f"❌ File not found: {args.file}")
            return

    if args.dir:
        if os.path.isdir(args.dir):
            files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                    if os.path.isfile(os.path.join(args.dir, f))]
            files_to_process.extend(files)
        else:
            logger.error(f"❌ Directory not found: {args.dir}")
            return

    if not files_to_process and not args.analyze:
        logger.error("❌ No files to process. Use --file or --dir argument.")
        parser.print_help()
        return

    # Process files
    if files_to_process:
        print(f"\n🎯 BATCH INGESTION")
        print(f"📂 Files to process: {len(files_to_process)}")

        results = []
        successful = 0
        failed = 0

        for i, file_path in enumerate(files_to_process):
            print(f"\n📁 FILE {i+1}/{len(files_to_process)}")
            result = process_and_ingest_document(
                file_path,
                client,
                config,
                doc_parser,
                content_processor
            )

            results.append(result)
            if result["success"]:
                successful += 1
            else:
                failed += 1

        # Final batch summary
        print("\n" + "="*70)
        print("🏆 BATCH INGESTION COMPLETE!")
        print("="*70)
        print(f"✅ Successfully processed: {successful} files")
        print(f"❌ Failed: {failed} files")

        total_chunks = sum(r.get("chunks", 0) for r in results if r["success"])
        total_time = sum(r.get("time", 0) for r in results if r["success"])

        print(f"📊 Total chunks ingested: {total_chunks}")
        print(f"⏱️  Total processing time: {total_time:.1f} seconds")

        if failed > 0:
            print(f"\n❌ Failed files:")
            for r in results:
                if not r["success"]:
                    print(f"   - {r['file']}: {r.get('error', 'Unknown error')}")

        print("="*70 + "\n")

    # Analyze collection if requested
    if args.analyze or files_to_process:
        analyze_collection(client, config)


if __name__ == "__main__":
    print("\n🧠 SEMANTIC CHUNKING DATA INGESTION")
    print("=" * 70)
    main()