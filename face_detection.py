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

class FaceDetection:
    def __init__(self):
        # Initialize the InsightFace model
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def generate_face_data(self, image_path):
        # Read the image
        face_data = []
        img_path_index_for_face = []
        for idx, image_path in enumerate(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image at path {image_path} could not be loaded.")
            
            # Convert BGR to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces and extract features
            faces = self.model.get(img)
            if len(faces) > 0:
                for face in faces:
                    face = dict(face)
                    face_data.append(face)
                    img_path_index_for_face.append(idx)
        
        face_data_with_labels = self.generate_clustering_face_labels(face_data)
        
        with open('embed_store/face_data.pkl', 'wb') as f:
            pickle.dump(face_data_with_labels, f)
        
        with open('embed_store/img_path_index_for_face.pkl', 'wb') as f:
            pickle.dump(img_path_index_for_face, f)

        return face_data
    
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


if __name__ == "__main__":
    # Example usage
    image_dir = "ImageSamples"  # Replace with the actual path
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    sample_image_path = image_paths
    
    face_detection = FaceDetection()
    face_data = face_detection.generate_face_data(sample_image_path)
    
    with open('embed_store/face_data.pkl', 'rb') as f:
        loaded_face_data = pickle.load(f)
    for face in loaded_face_data:
        print(face['label'])
    # face_data = face_detection.generate_clustering_face_labels(loaded_face_data)
    # for face in face_data:
    #     print(face['label'])