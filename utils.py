import faiss
import numpy as np
from PIL import Image

def get_next_faiss_id(index):
    current_ids = faiss.vector_to_array(index.id_map)
    return current_ids.max() + 1 if len(current_ids) > 0 else 0

def get_and_orient_image(image_path):
    try:
        image = Image.open(image_path)
        
        # Check for EXIF data
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            orientation_key = 274  # EXIF orientation tag
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                # Rotate/flip image based on orientation
                if orientation == 2:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180)
                elif orientation == 4:
                    image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    image = image.rotate(-90, expand=True)
                elif orientation == 7:
                    image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        
        # Convert to RGB and return
        return image.convert("RGB")
    except Exception as e:
        print(f"Could not re-orient image {image_path}: {e}")
        # Fallback: return image in RGB without orientation correction
        return Image.open(image_path).convert("RGB")