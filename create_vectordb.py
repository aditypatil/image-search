import os
import logging
import argparse
from typing import List
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import clip
import torch
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_image_paths(directory: str, number: int = None) -> List[str]:
    image_paths = []
    count = 1
    for filename in os.listdir(directory):
        # print(f"Filename : {filename}")
        if filename.endswith('.JPG'):
            image_paths.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return image_paths
            count += 1
    return image_paths

def get_features_from_image_path(image_paths):
    # print(f"Image Path : {image_paths}")
    device = "cuda" if torch.cuda.is_available()else ( "mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32",device=device) 
    images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    image_input = torch.tensor(np.stack(images))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_features

def index_images(dataset_folder):
    """Index images in the dataset folder."""
    # print(f"Dataset Folder : {dataset_folder}")
    image_paths = get_image_paths(dataset_folder,1000)
    image_embeddings = get_features_from_image_path(image_paths)
    image_embeddings = np.array(image_embeddings)
    return image_embeddings, image_paths

def create_vectordb(dataset_folder, vectordb_name):
    """Index images in the dataset folder and save to ChromaDB."""

    image_embeddings, image_paths = index_images(dataset_folder)
    # Index images in ChromaDB
    client = chromadb.PersistentClient(path=vectordb_name)

    image_loader = ImageLoader()
    multimodal_ef = OpenCLIPEmbeddingFunction()
    collection = client.get_or_create_collection(name="multimodal_db", embedding_function=multimodal_ef, data_loader=image_loader)
    for i, embedding in enumerate(image_embeddings):
        collection.add(
            ids=[str(i)],  # Ensure the ID is a string
            uris=image_paths[i],
            embeddings=[embedding],
            metadatas=[{"path": image_paths[i]}]
        )
    logger.info(f"Indexed {len(image_paths)} images into ChromaDB collection '{vectordb_name}'")

def main():
    parser = argparse.ArgumentParser(description="Index images in a dataset folder and save to ChromaDB.")
    parser.add_argument("dataset_folder", type=str, help="Path to the dataset folder containing images")
    parser.add_argument("vectordb_name", type=str, help="Name of the ChromaDB vector database")

    args = parser.parse_args()

    create_vectordb(args.dataset_folder, args.vectordb_name)

if __name__ == "__main__":
    main()