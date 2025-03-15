import streamlit as st
import os
import logging
from typing import List, Tuple
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB configurations
vectordb = os.getenv("vectordb") or "./chroma_db"  # Add default path if env var not set

# Load CLIP model for query embedding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model for image captioning
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def get_image_description(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        return description
    except Exception as e:
        logger.error(f"Error getting image description: {e}")
        return "Description unavailable"

def get_query_embedding(query_text: str) -> List[float]:
    """Generate CLIP embedding for the query text"""
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding.squeeze(0).tolist()

def search_images_chromadb(query_text: str, k: int, collection) -> List[Tuple[str, float]]:
    try:
        # Get query embedding using CLIP
        query_embedding = get_query_embedding(query_text)
        
        # Query the collection using the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'distances', 'metadatas', 'uris']
        )
        
        # Process and rank results
        ranked_results = []
        for i in range(len(results['uris'][0])):
            uri = results['uris'][0][i]
            distance = results['distances'][0][i]
            ranked_results.append((uri, distance))
        
        # Sort by distance (lower is better for similarity)
        ranked_results.sort(key=lambda x: x[1])
        
        return ranked_results[:k]
    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise

def find_top_k_images(query_text: str, index_name: str = "multimodal_db", k: int = 5) -> List[str]:
    """Retrieve top K similar images from ChromaDB."""
    try:
        # Load ChromaDB collection
        client = chromadb.PersistentClient(path=vectordb)
        image_loader = ImageLoader()
        multimodal_ef = OpenCLIPEmbeddingFunction()
        collection = client.get_or_create_collection(
            name=index_name,
            embedding_function=multimodal_ef,
            data_loader=image_loader
        )
        
        # Search for nearest neighbors
        results = search_images_chromadb(query_text, k, collection)
        return [result[0] for result in results]  # Return just the image paths
    except Exception as e:
        logger.error(f"Error in find_top_k_images: {e}")
        raise

# Streamlit app
st.title("Image Search with ChromaDB and CLIP")

query_text = st.text_input("Enter your query text:")
k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if query_text:
        try:
            top_k_results = find_top_k_images(query_text, k=k)
            for img_path in top_k_results:
                st.image(img_path, caption=get_image_description(img_path))
        except Exception as e:
            st.error(f"Error finding top k images: {e}")
    else:
        st.warning("Please enter a query text")
