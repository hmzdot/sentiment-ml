#!/bin/bash

# Start TensorBoard in the background
python -m tensorboard.main --logdir=/app/logs --host=0.0.0.0 --port=6006 &
TENSORBOARD_PID=$!

echo "TensorBoard started with PID $TENSORBOARD_PID"

# Run the training script
python src/train.py

# After training completes, stop TensorBoard
echo "Training finished. Stopping TensorBoard..."
kill $TENSORBOARD_PID
