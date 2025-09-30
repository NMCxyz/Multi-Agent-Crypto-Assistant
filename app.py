import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
# from elevenlabs.client import ElevenLabs  # Removed - speech features disabled

from config import Config
from agents.agent_decision import process_query
import importlib
from utils.mongodb_manager import get_mongodb_manager
from utils.memory_tools import get_memory_manager, get_conversation_history

# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Crypto/Financial Assistant", version="2.0")

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
# SPEECH_DIR = "uploads/speech"  # Commented out - speech features removed

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER]:  # Removed SPEECH_DIR
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# ElevenLabs client removed - speech features disabled
# client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# cleanup_old_audio function removed - speech features disabled
# def cleanup_old_audio():
#     """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
#     while True:
#         try:
#             files = glob.glob(f"{SPEECH_DIR}/*.mp3")
#             for file in files:
#                 os.remove(file)
#             print("Cleaned up old speech files.")
#         except Exception as e:
#             print(f"Error during cleanup: {e}")
#         time.sleep(300)  # Runs every 5 minutes

# Background cleanup thread removed - speech features disabled
# cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
# cleanup_thread.start()

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []

# SpeechRequest model removed - speech features disabled
# class SpeechRequest(BaseModel):
#     text: str
#     voice_id: str = "EXAMPLE_VOICE_ID"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}

@app.post("/reload")
def reload_modules():
    """Reload Python modules for development (requires restart=False in production)"""
    try:
        # Reload key modules
        importlib.reload(importlib.import_module("agents.agent_decision"))
        importlib.reload(importlib.import_module("config"))
        importlib.reload(importlib.import_module("utils.mongodb_manager"))
        importlib.reload(importlib.import_module("utils.memory_tools"))

        # Re-import process_query function
        from agents.agent_decision import process_query

        return {"status": "success", "message": "Modules reloaded successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to reload modules: {str(e)}"}
        )

@app.post("/chat")
def chat(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    """Process user text query through the multi-agent system with conversation history."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Process query with conversation history
    response_data = process_query(request.query, session_id=session_id)
    response_text = response_data['messages'][-1].content

    # Set session cookie
    response.set_cookie(key="session_id", value=session_id)

    # Prepare response result
    result = {
        "status": "success",
        "response": response_text,
        "agent": response_data["agent_name"]
    }

    return result

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """Process crypto/financial document uploads with optional text input."""
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )

    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )

    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    query = {"text": text, "image": file_path}
    response_data = process_query(query, session_id=session_id)
    response_text = response_data['messages'][-1].content

    # Set session cookie
    response.set_cookie(key="session_id", value=session_id)

    # Prepare response result
    result = {
        "status": "success",
        "response": response_text,
        "agent": response_data["agent_name"]
    }

    # Remove temporary file after sending
    os.remove(file_path)

    return result

@app.post("/validate")
def validate_crypto_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for crypto/financial AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Set session cookie
    response.set_cookie(key="session_id", value=session_id)

    # Re-run the agent decision system with the validation input
    validation_query = f"Validation result: {validation_result}"
    if comments:
        validation_query += f" Comments: {comments}"

    response_data = process_query(validation_query, session_id=session_id)

    if validation_result.lower() == 'yes':
        return {
            "status": "validated",
            "message": "**Analysis confirmed by financial expert:**",
            "response": response_data['messages'][-1].content
        }
    else:
        return {
            "status": "rejected",
            "comments": comments,
            "message": "**Analysis requires further review:**",
            "response": response_data['messages'][-1].content
        }

# Speech endpoints removed - speech features disabled
# @app.post("/transcribe")
# async def transcribe_audio(audio: UploadFile = File(...)):
#     """Endpoint to transcribe speech using ElevenLabs API"""
#     # ... (removed entire function)
#
# @app.post("/generate-speech")
# async def generate_speech(request: SpeechRequest):
#     """Endpoint to generate speech using ElevenLabs API"""
#     # ... (removed entire function)

@app.get("/conversations")
def get_conversations():
    """Get list of user conversations"""
    mongodb_manager = get_mongodb_manager()

    # Get conversation data from MongoDB
    db = mongodb_manager.db
    conversations_collection = db["conversations"]

    # Get all conversations sorted by creation time (newest first)
    conversations = list(conversations_collection.find(
        {},
        {"_id": 0, "session_id": 1, "title": 1, "created_at": 1, "message_count": 1}
    ).sort("created_at", -1).limit(50))

    # Format conversations for UI
    formatted_conversations = []
    for conv in conversations:
        formatted_conversations.append({
            "id": conv.get("session_id", ""),
            "title": conv.get("title", "Untitled Conversation")[:50] + "..." if len(conv.get("title", "")) > 50 else conv.get("title", "Untitled Conversation"),
            "last_message": conv.get("title", ""),  # Use title as last message for now
            "created_at": conv.get("created_at", ""),
            "message_count": conv.get("message_count", 0)
        })

    return {
        "conversations": formatted_conversations,
        "enabled": True,
        "total": len(formatted_conversations)
    }

@app.post("/conversations")
def create_conversation():
    """Create a new conversation"""
    mongodb_manager = get_mongodb_manager()

    # Generate new session ID
    new_session_id = str(uuid.uuid4())

    # Save conversation to MongoDB
    db = mongodb_manager.db
    conversations_collection = db["conversations"]

    from datetime import datetime
    conversation_data = {
        "session_id": new_session_id,
        "title": "New Conversation",
        "created_at": datetime.utcnow(),
        "message_count": 0,
        "last_updated": datetime.utcnow()
    }

    conversations_collection.insert_one(conversation_data)

    return {
        "success": True,
        "session_id": new_session_id,
        "message": "New conversation created"
    }

@app.post("/clear_session")
def clear_session(
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    """Clear current session memory cache"""
    # Import get_cache_manager properly
    from utils.mongodb_manager import get_mongodb_manager
    from utils.redis_cache import RedisCacheManager

    # Clear ALL Redis cache for this session (session_memory, prompt_response, etc.)
    if config.cache.enable_caching and session_id:
        cache_manager = RedisCacheManager()

        # Clear session_memory cache
        session_memory_key = f"session_memory:{session_id}"
        cache_manager.client.delete(session_memory_key)

        # Clear ALL prompt_response cache keys that contain this session_id
        all_keys = cache_manager.client.keys("prompt_response:*")
        session_keys = [key for key in all_keys if session_id in key]

        if session_keys:
            cache_manager.client.delete(*session_keys)

    # Generate new session ID
    new_session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=new_session_id)

    return {
        "success": True,
        "message": "Session cleared successfully",
        "new_session_id": new_session_id
    }

@app.delete("/conversations/{session_id}")
def delete_conversation(session_id: str):
    """Delete a conversation"""
    mongodb_manager = get_mongodb_manager()

    # Delete conversation from MongoDB
    db = mongodb_manager.db
    conversations_collection = db["conversations"]

    # Delete the conversation
    result = conversations_collection.delete_one({"session_id": session_id})

    if result.deleted_count > 0:
        return {
            "success": True,
            "message": f"Conversation {session_id} deleted"
        }
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Conversation {session_id} not found"}
        )

@app.get("/memory-info")
def get_memory_info():
    """Get memory system information"""
    mongodb_manager = get_mongodb_manager()
    memory_manager = get_memory_manager()

    # Get database info
    db_info = mongodb_manager.get_database_info()

    # Get conversation summaries
    conversation_summaries = memory_manager.get_relevant_memories(
        query="conversation summary",
        memory_type="conversation_summaries",
        limit=10
    )

    # Get user preferences
    user_preferences = memory_manager.get_relevant_memories(
        query="user preference",
        memory_type="user_preferences",
        limit=10
    )

    return {
        "enabled": mongodb_manager.is_enabled(),
        "database_info": db_info,
        "conversation_summaries": conversation_summaries,
        "user_preferences": user_preferences
    }

@app.post("/clear-memory")
def clear_memory():
    """Clear all memory data"""
    mongodb_manager = get_mongodb_manager()

    # Clear all memory collections
    db = mongodb_manager.db

    # Drop all memory-related collections
    collections_to_drop = [
        "memory_store",
        "thread_checkpoints"
    ]

    for collection_name in collections_to_drop:
        try:
            db[collection_name].drop()
        except Exception as e:
            print(f"Error dropping collection {collection_name}: {e}")

    return {"message": "Memory cleared successfully"}

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)