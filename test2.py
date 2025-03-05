import numpy as np
import os
import torch
import clip

#PART1
# Load CLIP model
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Using device:", device)


print("Clip model loaded sucessfully!")

import faiss

#PART2
# Set embedding dimension
d = 512  

# Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(d)  

print("FAISS index loaded successfully!")

#PART3
image_paths = []  # List to store image file paths
# Directory containing images
IMAGE_DIR = "/Users/adityapatil/Image_Search/sample"
os.makedirs(IMAGE_DIR, exist_ok=True)

print("Image directory defined successfully!")
