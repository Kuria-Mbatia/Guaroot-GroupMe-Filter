# bot.py
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from utils import send_message, handle_message
import uvicorn
from typing import Optional, List
import json

# Initialize FastAPI app
app = FastAPI(title="GroupMe Bot")

# Create thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

class Attachment(BaseModel):
    type: str
    url: Optional[str] = None

class MessageData(BaseModel):
    text: str
    user_id: str
    group_id: str
    id: str
    sender_id: str
    name: str
    sender_type: str
    attachments: List[Attachment] = []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "alive", "service": "GroupMe Bot"}

@app.post("/")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Main webhook endpoint for receiving GroupMe messages"""
    try:
        # Get raw data first
        data = await request.json()
        
        # Log incoming data for debugging
        print(f"Received webhook data: {json.dumps(data, indent=2)}")
        
        # Validate required fields
        if not all(key in data for key in ['text', 'user_id', 'group_id', 'id', 'sender_id']):
            return {"status": "error", "message": "Missing required fields"}

        # Process message asynchronously
        background_tasks.add_task(
            handle_message,
            message_text=data['text'],
            user_id=data['user_id'],
            group_id=data['group_id'],
            message_id=data['id'],
            sender_id=data['sender_id']
        )
        
        return {"status": "ok", "message": "Message queued for processing"}
    
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize any resources on startup"""
    print("Bot starting up...")
    send_message("Bot is online and ready to process messages!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    print("Bot shutting down...")
    executor.shutdown(wait=True)
    send_message("Bot is going offline for maintenance.")

def start_bot(host="0.0.0.0", port=5000):
    """Start the bot with uvicorn server"""
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            workers=1  # Adjust based on your needs
        )
    except Exception as e:
        print(f"Failed to start bot: {str(e)}")

if __name__ == "__main__":
    start_bot()