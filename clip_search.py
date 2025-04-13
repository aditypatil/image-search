import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

class CLIPSearch:
    def __init__(self, model_name="openai/clip-vit-base-patch32", embedding_dir="embed_store"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.embedding_dir = embedding_dir
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()
        os.makedirs(self.embedding_dir, exist_ok=True)

    def generate_embeddings(self, image_paths):
        embeddings = []
        img_filenames = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy())
            img_filenames.append(os.path.basename(image_path))
        
        np.save(f"{self.embedding_dir}/img_embeddings.npy", embeddings)  
        np.save(f"{self.embedding_dir}/img_filenames.npy", img_filenames)  
        
        pass

    def search(self, query, top_k=5):
        query_inputs = self.tokenizer([query], return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**query_inputs)
        text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        embedding_files = "img_embeddings.npy"
        img_embeddings = np.load(os.path.join(self.embedding_dir, embedding_files), allow_pickle=True)
        img_filenames = np.load(os.path.join(self.embedding_dir, "img_filenames.npy"), allow_pickle=True)

        similarities = []
        for idx, embedding in enumerate(img_embeddings):
            embedding = embedding.reshape(1, -1)
            similarity = cosine_similarity(text_embeddings.cpu().numpy(), embedding)[0]
            similarities.append(similarity)
        
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i][0], reverse=True)
        
        top_indices = sorted_indices[:top_k]
        # Retrieve filenames and similarity scores
        results = [(img_filenames[i], similarities[i]) for i in top_indices]

        return results