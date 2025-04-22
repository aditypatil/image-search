import os
import faiss
import pickle
import numpy as np
import models.clip_search as clip_search
import models.face_detection as face_detection
import models.geo_metadata as geo_metadata

class Search:

    def __init__(self, load_indices = True, embed_dir = 'embed_store'):

        self.embed_dir = embed_dir

        # loading up the indices from embed_dir
        self.img_path_index = pickle.load(open(os.path.join(self.embed_dir, 'img_path_index.pkl'), 'rb'))
        self.face_data = pickle.load(open(os.path.join(self.embed_dir, 'face_data.pkl'), 'rb'))
        self.flatten_img_face_index = pickle.load(open(os.path.join(self.embed_dir, 'img_path_index_for_face.pkl'), 'rb'))
        self.clip_embed = faiss.read_index(os.path.join(self.embed_dir, 'img_embeddings.bin'))
        self.geo_data = np.load(os.path.join(self.embed_dir, 'geo_metadata.npy'), allow_pickle=True)
    
    def strategy1(self, query):

        # geo indices
        searchgeo = geo_metadata.SearchBM25()
        query = "goa"
        G_indices = searchgeo.search(query)

        # face indices to get F_indices


        # strict sequential search strategy
        F_set = set(F_indices)
        G_set = set(G_indices)

        # Apply the conditional logic
        if F_set and G_set:  # True if both sets are non-empty
            # Condition 1 met (both have results): Take intersection
            combined_indices = F_set.intersection(G_set)
        else:
            # Condition 2 met (at least one is empty, or both are empty): Take union
            combined_indices = F_set.union(G_set)


        # clip search on combined_indices. If combined indices blank, then search through entire index. Else, search into index only on images searched by combined_indices


        
        
        return 



def __main__():

    # load up the embed_store
    # first search strategy: F, G then CLIP:
        # IF F&G both retrieve indices: intersection
        # if either retrieve indices with other being null: union
        # use the indices retrieved from both (after intersection or union) then look into CLIP
    
    img_indices = Search().strategy1(query = "goa")
    print(img_indices)


    pass

if __name__ == '__main__':
    __main__()
