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

## 4. Evaluate sentiment for any text

# Either with uv
uv run src/eval.py out/{model_name} {text}

# Or with python3 (assuming venv is active)
python3 src/eval.py out/{model_name} {text}
```