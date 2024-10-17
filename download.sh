#!/bin/bash

# Create weights directory if it doesn't exist
mkdir -p weights

# Download the weights from the GitHub release and save to weights folder
wget -O weights/faceboxes.pth https://github.com/yakhyo/faceboxes-pytorch/releases/download/v0.0.1/faceboxes.pth

echo "Download complete. Weights saved to 'weights/faceboxes.pth'."