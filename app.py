from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import sys
import uvicorn
import logging
from data_loader import HospitalDataLoader
from chat import retriever, chat

# Setup console stream for logging
console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log", encoding='utf-8'),
        logging.StreamHandler(console_stream)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Data Chat API",
    description="API for hospital data retrieval and chat functionality",
    version="1.0.0"
)

# Mount the 'static' directory to serve CSS, JS, and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data loader and retriever
try:
    data_loader = HospitalDataLoader()
    retriever.refresh_indexes()
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    raise

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse(
    request=request,
    name="indexa.html",
    context={}
)

@app.post("/refresh_data")
async def refresh_data_endpoint():
    """Refresh hospital data and retrieval models"""
    try:
        logger.info("Hospital data refresh request received.")
        global data_loader
        data_loader = HospitalDataLoader()
        retriever.refresh_indexes()
        logger.info("Hospital data and retrieval models refreshed successfully.")
        return {"message": "Hospital data and retrieval models refreshed successfully."}
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh data")

@app.get("/api/metadata-tags")
async def get_metadata_tags():
    """Get all metadata tags"""
    try:
        tags = data_loader.get_all_metadata_tags()
        return {"tags": tags}
    except Exception as e:
        logger.error(f"Error fetching metadata tags: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch metadata tags")

@app.get("/api/tag-counts")
async def get_metadata_tag_counts():
    """Get counts for all metadata tags"""
    try:
        return {"tag_counts": data_loader.get_metadata_tag_counts()}
    except Exception as e:
        logger.error(f"Error fetching tag counts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch tag counts")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: Request, chat_data: ChatInput, x_user_id: str = Header(None)):
    """Handle chat messages"""
    try:
        user_message = chat_data.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        user_id = x_user_id or request.client.host
        response = chat(user_message, user_id)
        logger.info(f"Chat response generated for user {user_id}")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")
    
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=False, log_level="info")