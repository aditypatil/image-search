import os
import torch
import clip
import numpy as np
from PIL import Image
from fastapi import FastAPI, Query
from typing import List
from torchvision import transforms

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Clip model loaded sucessfully!")

import faiss
# Initialize FAISS index (for storing embeddings)
d = 512  # CLIP embedding dimension
index = faiss.IndexFlatL2(d)

print("FAISS index loaded successfully!")

image_paths = []  # List to store image file paths
# Directory containing images
IMAGE_DIR = "/Users/adityapatil/Image_Search/sample"
os.makedirs(IMAGE_DIR, exist_ok=True)

print("Image directory defined successfully!")


# Function to index images
def index_images():
    global image_paths
    image_paths.clear()
    features = []
    
    for img_file in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()
        
        features.append(embedding)
        image_paths.append(img_path)
    
    if features:
        features = np.vstack(features)
        index.add(features)
        print(f"Indexed {len(features)} images.")
    else:
        print("No images found.")

# Index images initially
index_images()

# FastAPI app setup
app = FastAPI()

@app.get("/search/")
def search_images(query: str = Query(..., description="Search query")):
    text = clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text).cpu().numpy()
    
    distances, indices = index.search(text_embedding, k=5)
    results = [image_paths[i] for i in indices[0] if i < len(image_paths)]
    
    return {"query": query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
