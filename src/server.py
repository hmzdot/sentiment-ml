from fastapi import FastAPI
from eval import eval
import os

app = FastAPI()

SNAPSHOT_DIR = "snapshots"


def get_latest_snapshot():
    snapshots = filter(lambda x: x.endswith(".pth"), os.listdir(SNAPSHOT_DIR))
    if not snapshots:
        return None
    return sorted(snapshots)[-1]


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}


@app.post("/evaluate/")
def evaluate(text: str):
    latest_snapshot = get_latest_snapshot()
    if not latest_snapshot:
        return {"error": "No snapshots found."}

    snapshot_path = os.path.join(SNAPSHOT_DIR, latest_snapshot)
    sentiment = eval(snapshot_path, text)
    prediction = "Positive" if sentiment > 0.5 else "Negative"

    return {
        "text": text,
        "sentiment_score": sentiment,
        "prediction": prediction,
    }
