import os
import sys
import torch
import clip
import chromadb
import numpy as np
from PIL import Image, ExifTags

# Load CLIP model and set device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/16", device=device)

# Initialize ChromaDB client and collection with explicit cosine distance
chroma_client = chromadb.PersistentClient(path="./chroma_db1")
collection = chroma_client.get_or_create_collection(name="image_embeddings", metadata={"hnsw:space": "cosine"})

SIMILARITY_CUTOFF = 0.25  # Minimum similarity threshold


def get_text_embedding(text_query):
    """
    Extracts CLIP embedding for a text query.
    """
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text_query]).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
    return text_features.cpu().numpy()[0].tolist()


def search_similar_images(text_query, top_k=5):
    """
    Searches for the most similar images in ChromaDB based on a text query.
    """
    text_embedding = get_text_embedding(text_query)
    results = collection.query(query_embeddings=[text_embedding], n_results=top_k)
    
    if "ids" not in results or not results["ids"]:
        print("No matching images found in ChromaDB.")
        return [], []
    
    ids = results["ids"][0]
    distances = results["distances"][0]
    paths = [doc["path"] for doc in results["metadatas"][0]]
    
    # Convert distances to similarity scores (cosine distance = 1 - cosine similarity)
    similarities = [1 - dist for dist in distances]
    
    # Apply similarity cutoff
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= SIMILARITY_CUTOFF]
    if not filtered_indices:
        print("No image with the description found.")
        print("Would you be interested in seeing other remotely similar images?")
        return paths[:2], similarities[:2]  # Return top 2 closest matches if all are below threshold
    
    return [os.path.basename(paths[i]) for i in filtered_indices], [similarities[i] for i in filtered_indices]


# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_images.py \"<query>\"")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 5
    image_filenames, similarities = search_similar_images(query, top_k=top_k)

    if image_filenames:
        print("Top matching images:")
        for filename, sim in zip(image_filenames, similarities):
            print(f"{filename} - Similarity: {sim:.4f}")
