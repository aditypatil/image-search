import numpy as np
import pickle
import os
from collections import Counter
from typing import List, Tuple
import pandas as pd

class PeopleLibrary:
    def __init__(self, embedding_dir="embed_store"):
        self.embedding_dir = embedding_dir
        self.index_file = 'img_path_index_for_face.pkl'
        self.face_file = 'face_data.pkl'
        # facebook: List[dict]
    
        with open(os.path.join(self.embedding_dir, self.index_file), 'rb') as f:
            self.img_path_index = pickle.load(f)
        with open(os.path.join(self.embedding_dir, self.face_file), 'rb') as f:
            self.facebook = pickle.load(f)
        
    def _get_top_faces(self, top_k=5):

        image_frequency = Counter(self.img_path_index)
        face_frequency = Counter([int(face['label']) for face in self.facebook])
        top_k_faces = [label for label, _ in face_frequency.most_common(top_k)]
        top_faces = []
        for face in top_k_faces:
            filtered = [fb for fb in self.facebook if fb['label'] == face]
            max_entry = max(filtered, key=lambda x: x['det_score'])
            top_faces.append(max_entry)
        return top_faces


if __name__=="__main__":
    people = PeopleLibrary()

    plist = people._get_top_faces(top_k=10)
    print([(int(face['label']), face['bbox']) for face in plist])
    