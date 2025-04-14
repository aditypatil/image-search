import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from pillow_heif import register_heif_opener
import faiss
from utils import get_next_faiss_id, get_and_orient_image
import json
import time


class CLIPSearch:
    def __init__(self, model_name="openai/clip-vit-base-patch32", embedding_dir="embed_store", embed_size=512):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.embedding_dir = embedding_dir
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.embed_size = embed_size
        self.next_id = 0
        self.img_index = None
        self.img_index_wrap = None
        
        print(f"Using device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()
        os.makedirs(self.embedding_dir, exist_ok=True)

        self.mapping_file = os.path.join(self.embedding_dir, "image_id_to_path.json")
        self.index_file = os.path.join(self.embedding_dir, "image_embeddings.faiss")
        self.index_wrap_file = os.path.join(self.embedding_dir, "image_enbeddings_indexed.faiss")
        
        try:
            self.img_index = faiss.read_index(self.index_file)
            self.img_index_wrap = faiss.read_index(self.index_wrap_file)
            print(f"Loaded existing index with {self.img_index.ntotal} embeddings")
        except Exception as e:
            print(f"Creating new FAISS index: {str(e)}")
            self.img_index = faiss.IndexFlatIP(self.embed_size)
            self.img_index_wrap = faiss.IndexIDMap(self.img_index)
        self.next_id = get_next_faiss_id(self.img_index_wrap)
            
        # Try to load the image mapping
        try:
            with open(self.mapping_file, "r") as f:
                self.img_mapping = json.load(f)
                self.img_mapping = {int(k): v for k, v in self.img_mapping.items()}
            print("Loaded existing image ID mapping")
        except:
            print("Image ID mapping not found, creating new empty dictionary.")
            self.img_mapping = {}

    def generate_embeddings(self, image_paths):
        for image_path in image_paths:
            image = get_and_orient_image(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()

            self.img_index_wrap.add_with_ids(image_features.reshape(1, -1), np.array([self.next_id], dtype=np.int64))
            self.img_mapping[self.next_id] = image_path
            self.next_id += 1

        with open(self.mapping_file, "w") as f:
            json.dump(self.img_mapping, f, indent=4)
        
        faiss.write_index(self.img_index, self.index_file)
        faiss.write_index(self.img_index_wrap, self.index_wrap_file)
        
        print(f"Sucessfully indexed {self.img_index.ntotal} images")
        pass

    def search(self, query, top_k=5):
        query_inputs = self.tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**query_inputs)
            text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_embeddings = np.array(text_embeddings.cpu().numpy(), dtype='float32')        
        similarities, indices = self.img_index.search(text_embeddings, top_k)

        matched_paths = [self.img_mapping.get(int(idx), None) for idx in indices[0]]
        results = [[path, float(similarity)] for path, similarity in zip(matched_paths, similarities[0])
               if path is not None]
        
        return results