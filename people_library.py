import numpy as np
import pickle
import os
from collections import Counter
from typing import List, Tuple
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_and_orient_image

class PeopleLibrary:
    def __init__(self, embedding_dir="embed_store"):
        self.embedding_dir = embedding_dir
        self.index_file = 'img_path_index_for_face.pkl'
        self.face_file = 'face_data.pkl'
        self.original_facebook = None
        # facebook: List[dict]
    
        with open(os.path.join(self.embedding_dir, self.index_file), 'rb') as f:
            self.img_index = pickle.load(f)
        with open(os.path.join(self.embedding_dir, self.face_file), 'rb') as f:
            self.original_facebook = pickle.load(f)
        self.facebook = self.original_facebook
        with open(os.path.join(self.embedding_dir, 'image_id_to_path.json'), 'rb') as f:
            self.img_path_map = json.load(f)
        # Push image index into facebook for image path tracking
        for i, face in enumerate(self.facebook):
            face['img_index'] = self.img_index[i]
            face['img_path'] = self.img_path_map[str(self.img_index[i])]
        
        self.__name_column_check()
        
    def __name_column_check(self):
        if 'name' not in [key for key, _ in self.original_facebook[-1].items()]:
            for i in range(len(self.original_facebook)):
                self.original_facebook[i]['name'] = ''


    def _get_top_faces(self, top_k=20):
        face_frequency = Counter([int(face['label']) for face in self.facebook])
        top_k_faces = [label for label, _ in face_frequency.most_common(top_k)]
        top_faces = []
        for face in top_k_faces:
            filtered = [fb for fb in self.facebook if fb['label'] == face]
            max_entry = max(filtered, key=lambda x: x['det_score'])
            top_faces.append(max_entry)
        return top_faces
    
    def _get_face_crop(self, face):
        img = get_and_orient_image(face['img_path'])
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        # Add some padding around the face (20%)
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        # Apply padding but stay within image bounds
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(img.width, x2 + padding_x)
        y2 = min(img.height, y2 + padding_y)
        # Crop the face
        face_img = img.crop((x1, y1, x2, y2))
        plt.figure(figsize=(1, 1))
        plt.imshow(face_img)
        plt.axis("off")
        plt.title(face['label'])
        plt.show()

    def rename_face(self, label, name):
        for face in self.original_facebook:
            if face['label']==label:
                face['name'] = name
    
    def write_out(self):
        with open(os.path.join(self.embedding_dir, self.face_file), 'wb') as f:
            pickle.dump(self.original_facebook, f)

if __name__=="__main__":
    people = PeopleLibrary()
    # get list of top k faces
    plist = people._get_top_faces(top_k=2)
    print([f"Face Cluster: {int(face['label'])}, Name: {face['name']}, Image Index: {face['img_index']}, Image Path: {face['img_path']} BBox: {face['bbox']}" for face in plist])
    # get the face corresponding to these faces
    people._get_face_crop(plist[0])
    # add feature to present this in frontend

    # rename faces
    people.rename_face(plist[0]['label'], 'Aditya')
    print([f"Face Cluster: {int(face['label'])}, Name: {face['name']}, Image Index: {face['img_index']}, Image Path: {face['img_path']} BBox: {face['bbox']}" for face in plist])
    # save changes to original database
    people.write_out()



    