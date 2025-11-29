from main import app
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST"), port=int(os.getenv("PORT")))