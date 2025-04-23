import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import faiss
from tqdm import tqdm


class CLIP:
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

class CLIPSearch:

    def __init__(self, clip_embeddings, subset_id = None, model_name="openai/clip-vit-base-patch32"):
        self.subset_id = subset_id
        self.clip_embeddings = clip_embeddings

        self.model = CLIPModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
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
    

    def search_faiss(self, query, top_k=5):
        query_inputs = self.tokenizer([query], return_tensors="pt")

        with torch.no_grad():
            text_features = self.model.get_text_features(**query_inputs)
        text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # embedding_files = "img_embeddings.bin" # currently looading from a single .bin file
        img_embeddings = self.clip_embeddings
        # img_filenames = np.load(os.path.join(self.embedding_dir, "img_filenames.npy"), allow_pickle=True)

        if self.subset_id is not None:
            id_selector = faiss.IDSelectorBatch(np.array(self.subset_id, dtype='int64'))
            search_params = faiss.SearchParameters()
            search_params.sel = id_selector

            D, I = img_embeddings.search(text_embeddings, k = 5, params=search_params)
        
        else:
            D, I = img_embeddings.search(text_embeddings, k=5) # IndexFlatL2 search for text embeddings

        return I[0]


def __main__():
    
    # import pickle

    # clip = CLIPSearch()

    # query = "mountains"

    # query_inputs = clip.tokenizer([query], return_tensors="pt")

    # with torch.no_grad():
    #     text_features = clip.model.get_text_features(**query_inputs)
    # text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)



    # # Load the index from the file
    # with open(os.path.join("embed_store", "img_path_index.pkl"), "rb") as f:
    #     img_path_index = pickle.load(f)
    
    # index = faiss.read_index('embed_store/img_embeddings.bin')

    # subset_ids = [0, 2, 5, 7, 10, 12, 17]

    # # Create an IDSelector to specify the subset of vectors
    # id_selector = faiss.IDSelectorBatch(np.array(subset_ids, dtype='int64'))

    # # Create Search Parameters and set the IDSelector
    # search_params = faiss.SearchParameters()
    # search_params.sel = id_selector

    # # Perform the search with the IDSelector
    # D, I = index.search(text_embeddings, k = 5, params=search_params)
    # print(D, I)


    # Dfull, Ifull = index.search(text_embeddings, k = 5)
    # print(Dfull, Ifull)
    
    # selected_paths = [img_path_index[i] for i in Ifull[0]]
    # print(selected_paths)

    pass

if __name__ == "__main__":
    __main__()