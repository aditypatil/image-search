import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import faiss
from tqdm import tqdm

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
        for image_path in tqdm(image_paths):
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
    
    def generate_embeddings_faiss(self, image_paths):
        embeddings = []
        img_filenames = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy())
            img_filenames.append(os.path.basename(image_path))
        
        # create faiss index
        embeddings_index = faiss.IndexFlatL2(embeddings[0].shape[1])

        # add embeddings to the index and saving as .bin file
        embeddings_index.add(np.array(embeddings).reshape(-1, embeddings[0].shape[1]))
        faiss.write_index(embeddings_index, f"{self.embedding_dir}/img_embeddings.bin")
        # saving file names as .npy file
        np.save(f"{self.embedding_dir}/img_filenames.npy", img_filenames)  
        
        pass

    def search_faiss(self, query, top_k=5):
        query_inputs = self.tokenizer([query], return_tensors="pt")

        with torch.no_grad():
            text_features = self.model.get_text_features(**query_inputs)
        text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        embedding_files = "img_embeddings.bin" # currently looading from a single .bin file
        img_embeddings = faiss.read_index(os.path.join(self.embedding_dir, embedding_files))
        img_filenames = np.load(os.path.join(self.embedding_dir, "img_filenames.npy"), allow_pickle=True)

        D, I = img_embeddings.search(text_embeddings, k=5) # IndexFlatL2 search for text embeddings

        results = [(img_filenames[idx], D[0][i]) for i, idx in enumerate(I[0])] # D[0] and I[0] since query is 1D

        return results


def __main__():

    # import inspect
    # clip_search = CLIPSearch()
    # print(inspect.signature(faiss.IndexFlatL2.search).parameters)
    # print(dir(faiss.IDSelectorBatch))
    # results = clip_search.search_faiss('pink flower', top_k=5)
    # print(results)
    # img_embed, img_filename, text_embed = clip_search.search_faiss("pink flower")
    # print(img_embed.ntotal, img_filename.shape, text_embed.shape)


    # indices_to_search = [0, 1, 2, 3, 4, 10, 12, 14]  # Example indices to search
    # ind_to_search_2 = np.random.choice(100, 20, replace=False)
    # params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(ind_to_search_2, dtype='int64')))
    
    # D, I = img_embed.search(text_embed, k=5, params = params) # IndexFlatL2 search for text embeddings
    # print(D, I)

    # D, I = img_embed.search(text_embed, k=5)
    # print(D, I)

    # results = [(img_filename[idx], D[0][i]) for i, idx in enumerate(I[0])]
    # print(results)
    # image_dir = "ImageSamples"  # Replace with the actual path
    # image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

    # clip_search.generate_embeddings_faiss(image_paths[:10])

    # embed_idx = faiss.read_index("embed_store/img_embeddings.bin")
    # embed_idx.search()

    # query = "A description of the image you want to search for"
    # results = clip_search.search(query, top_k=5)
    # for filename, score in results:
    #     print(f"Filename: {filename}, Similarity Score: {score}")

    pass

if __name__ == "__main__":
    __main__()