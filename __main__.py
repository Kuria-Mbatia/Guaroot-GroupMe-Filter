import threading
from bot import app
from push_service import start_push_service
import uvicorn
from utils import initialize_bot

if __name__ == "__main__":
    # Initialize bot settings
    initialize_bot()
    
    # Start the push service in a separate thread
    push_thread = threading.Thread(target=start_push_service)
    push_thread.daemon = True
    push_thread.start()
    
    # Run the FastAPI application with uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")