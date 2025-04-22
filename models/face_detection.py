import faiss
import insightface
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import faiss
import insightface
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class FaceDetection:
    def __init__(self, embedding_dir="embed_store"):
        # Initialize the InsightFace model
        self.model = insightface.app.FaceAnalysis()
        self.embedding_dir = embedding_dir
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def generate_face_data(self, image_path):
        '''
        Generate face data from images in the given directory.
        Adds labels to the face data using clustering.
        saves the face data and image path index to disk.
        '''

        # Read the image
        face_data = []
        img_path_index_for_face = []
        for idx, image_path in tqdm(enumerate(image_path), total=len(image_path)):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image at path {image_path} could not be loaded.")
            
            # Detect faces and extract features
            faces = self.model.get(img)
            if len(faces) > 0:
                for face in faces:
                    face = dict(face)
                    face_data.append(face)
                    img_path_index_for_face.append(idx)
        
        face_data_with_labels = self.generate_clustering_face_labels(face_data)
        
        with open(os.path.join(self.embedding_dir, 'face_data.pkl'), 'wb') as f:
            pickle.dump(face_data_with_labels, f)
            
        with open(os.path.join(self.embedding_dir, 'img_path_index_for_face.pkl'), 'wb') as f:
            pickle.dump(img_path_index_for_face, f)

        pass
    
    def generate_clustering_face_labels(self, face_data):

        normalized_face_embeds = np.array([face['embedding'] / np.linalg.norm(face['embedding']) for face in face_data])

        clustering_estimator = sklearn.cluster.AgglomerativeClustering(metric='cosine', 
                                                          linkage='average', 
                                                          distance_threshold=0.5,
                                                          n_clusters=None)
        
        labels = clustering_estimator.fit_predict(normalized_face_embeds)

        for idx, face in enumerate(face_data):
            face['label'] = labels[idx]

        return face_data
    
    def search(self, query):
        '''
        Search for images based on a query string using BM25 algorithm.
        returns a list of image path indices that match the query.
        '''

        # Load the face data and image path index
        with open('embed_store/face_data.pkl', 'rb') as f:
            loaded_face_data = pickle.load(f)
        
        with open('embed_store/img_path_index_for_face.pkl', 'rb') as f:
            img_path_index_for_face = pickle.load(f)

        #Create corpus/docs for BM25 tokenization
        name_labels = []
        for face in loaded_face_data:
            name_labels.append(face['label'])
        
        tokenized_corpus = [" " if isinstance(doc, np.int64) else doc.lower().split(" ") for doc in name_labels]
        bm25 = BM25Okapi(tokenized_corpus)

        # tokenize the search query
        tokenized_query = query.lower().split(" ")
        img_label_scores = bm25.get_scores(tokenized_query)

        retrieved_img_indices = []
        for idx, score in enumerate(img_label_scores):
            if score > 0:
                retrieved_img_indices.append(img_path_index_for_face[idx])

        return retrieved_img_indices


def __main__():
    # Example usage
    # image_dir = "ImageSamples"  # Replace with the actual path
    # image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    # sample_image_path = image_paths
    
    # face_detection = FaceDetection()
    # face_data = face_detection.generate_face_data(sample_image_path)
    # img_indices = face_detection.search(query="bigboig")
    # print(img_indices)
    # with open('embed_store/face_data.pkl', 'rb') as f:
    #     loaded_face_data = pickle.load(f)


    # name_labels = {0: "Aditya", 1: "Aditi", 2: "bigboig", 3: "macdriller", 4: "ineedfood"}
    # for idx, face in enumerate(loaded_face_data):
    #     if face['label'] in name_labels:
    #         face['label'] = name_labels[face['label']]
    
    # labels = []
    # for face in loaded_face_data:
    #     labels.append(face['label'])
    # print(labels)

    # with open('embed_store/face_data.pkl', 'wb') as f:
    #     pickle.dump(loaded_face_data, f)
    # for face in loaded_face_data:
    #     print(face['label'])
    # face_data = face_detection.generate_clustering_face_labels(loaded_face_data)
    # for face in face_data:
    #     print(face['label'])

    pass

if __name__ == "__main__":
    __main__()