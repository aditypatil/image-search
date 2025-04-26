
import os
import pickle
import models.clip_search as clip_search
import models.face_detection as face_detection
import models.geo_metadata as geo_metadata
import people_library


# handle for all data ingestion cases
# generate image paths from directory and store list. This will be index for all other metadata ingestion
# implementation: ingestion.generate_img_metadata() --> generates clip embeddings, face detection labels, geo metadata. 
# all to share common index of image_path
# ingestion will also deal with UI

# from clip search.py: generate embeddings for faiss and store
# from face detection.py: generate face data and store

class Ingestion:
    def __init__(self, embedding_dir="embed_store", images_dir = "ImageSamples"):
        self.embedding_dir = embedding_dir
        self.images_dir = images_dir
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _img_path_index(self):
        # Generate image paths from directory and store list
        formats = ('.jpg', '.jpeg', '.png', '.heic', '.heif')
        image_paths = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.lower().endswith(formats)]
        # save image paths to disk
        with open(os.path.join(self.embedding_dir, 'img_path_index.pkl'), 'wb') as f:
            pickle.dump(image_paths, f)
        return image_paths

    def run_pipeline(self):
        # Generate image paths from directory and store list

        # load image paths from disk if already generated
        if not os.path.exists(os.path.join(self.embedding_dir, 'img_path_index.pkl')):
            image_paths = self._img_path_index()
        else:
            with open(os.path.join(self.embedding_dir, 'img_path_index.pkl'), 'rb') as f:
                image_paths = pickle.load(f)

        # INITIALISE CLIP SEARCH AND FACE DETECTION MODELS
        clip = clip_search.CLIP(embedding_dir=self.embedding_dir)
        faceDetection = face_detection.FaceDetection(embedding_dir=self.embedding_dir)
        geo = geo_metadata.GeoExtractor(agent='my_agent')

        #GENERATE AND STORE METADATA 
        clip.generate_embeddings_faiss(image_paths=image_paths)
        faceDetection.generate_face_data(image_path=image_paths)
        geo.generate_geo_metadata(image_paths=image_paths)

        
        #Placeholder to run the label naming on face data
        people = people_library.PeopleLibrary()
        plist = people._get_top_faces(top_k=20)
        for idx, p in enumerate(plist):
            print(f"img path index: {idx}")
            people._get_face_crop(p)
        
        return plist

def __main__():

    data_ingestion = Ingestion(embedding_dir='embed_store', images_dir="ImageSamples")
    plist = data_ingestion.run_pipeline()

    return plist

if __name__ == "__main__":
    __main__()

        