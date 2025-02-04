# Sentiment

Start-to-production implementation of sentiment analysis with PyTorch

## Install and Run
With `uv` installed
```sh
## 1. Clone the repo

git clone git@github.com:hmzdot/sentiment-ml.git
cd sentiment-ml

## 2. Install dependencies

# Either with uv
uv sync

# Or with pip (assuming you've setup venv yourself)
pip3 install .

## 3. Train the model

# Either with uv
uv run src/train.py

# Or with python3 (assuming venv is active)
python3 src/train.py

# Or using docker
docker build -t sentiment-train -f dockerfiles/train.Dockerfile .
docker run --rm sentiment-train

## 4a. Evaluate sentiment for any text

# Either with uv
uv run src/eval.py snapshots/{model_name} {text}

# Or with python3 (assuming venv is active)
python3 src/eval.py snapshots/{model_name} {text}

# Or using docker
docker build -t sentiment-eval -f dockerfiles/eval.Dockerfile .
docker run --rm sentiment-eval snapshots/{model_name} {text} 

## 4b. Run evaluation server
uvicorn src.server:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/evaluate/?text=hello
# {"text":"hello","sentiment_score":"0.67","prediction":"Positive"}
```