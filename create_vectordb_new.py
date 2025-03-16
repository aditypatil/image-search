import os
import torch
import clip
import chromadb
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load CLIP model and set device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/16", device=device)

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db1")
collection = chroma_client.get_or_create_collection(name="image_embeddings", metadata={"hnsw:space": "cosine"})


def get_image_embeddings(image_paths):
    """
    Extracts CLIP embeddings for a list of images and stores them in ChromaDB if they don't already exist.
    """
    batch_size = 32
    
    # Fetch existing image IDs from the database
    existing_ids = set(doc["id"] for doc in collection.get()["documents"] or [])
    new_images = [img_path for img_path in image_paths if os.path.basename(img_path) not in existing_ids][:1000]
    
    if not new_images:
        print("All images are already stored in ChromaDB. No new embeddings added.")
        return
    
    for i in tqdm(range(0, len(new_images), batch_size), desc="Embedding Images"):
        batch_paths = new_images[i:i+batch_size]
        images = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in batch_paths]
        image_inputs = torch.cat(images)
        
        with torch.no_grad():
            features = model.encode_image(image_inputs)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        features_np = features.cpu().numpy()
        
        # Store new embeddings in ChromaDB
        for j, img_path in enumerate(batch_paths):
            collection.add(
                ids=[os.path.basename(img_path)],  # Store filename as ID
                embeddings=[features_np[j].tolist()],
                metadatas=[{"path": img_path}]
            )
    
    print(f"Added {len(new_images)} new image embeddings to ChromaDB.")


# Example usage
image_dir = "/Users/adityapatil/photos_backup"
image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith((".jpg", ".png", ".jpeg"))]

# Store image embeddings in ChromaDB if not already stored
get_image_embeddings(image_paths)

print("Embedding process completed!")