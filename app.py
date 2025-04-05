import logging
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download VADER lexicon
nltk.download('vader_lexicon')

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Validate YouTube API key
if not youtube_api_key:
    logger.error("Missing YouTube API key in .env")
    raise RuntimeError("YouTube API key missing.")

# Initialize YouTube API
try:
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    logger.debug("YouTube API initialized.")
except Exception as e:
    logger.exception("YouTube init failed")
    raise RuntimeError(f"YouTube API error: {e}")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    """Remove URLs, mentions, and special characters from text"""
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment using VADER"""
    scores = analyzer.polarity_scores(clean_text(text))
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def fetch_youtube_comments(video_id: str, count: int):
    """Fetch comments from YouTube video"""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=min(count, 100)  # YouTube's max per request
        )
        response = request.execute()
        
        comments = []
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item["id"],
                "text": comment["textDisplay"],
                "created_at": comment["publishedAt"]
            })
        return comments
        
    except HttpError as e:
        if e.resp.status == 403:
            raise HTTPException(status_code=400, detail="Comments are disabled for this video")
        logger.exception("YouTube API error")
        raise HTTPException(status_code=500, detail="YouTube API error")
    except Exception as e:
        logger.exception("Failed to fetch YouTube comments")
        raise HTTPException(status_code=500, detail=f"Error fetching comments: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "has_twitter": False,  # Flag for template to hide Twitter options
        "has_youtube": True
    })

@app.post("/analyze/")
async def analyze_youtube(
    request: Request,
    video_id: str = Form(..., min_length=11, max_length=11),  # Standard YouTube ID length
    count: int = Form(..., ge=1, le=100)  # Validate count between 1-100
):
    """Analyze YouTube video comments"""
    try:
        comments = fetch_youtube_comments(video_id.strip(), count)
        results = []
        for comment in comments:
            results.append({
                **comment,
                "sentiment": analyze_sentiment(comment["text"]),
                "created_at": comment["created_at"]
            })
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )