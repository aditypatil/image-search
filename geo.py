import numpy as np
import math
import os
from geopy.geocoders import Nominatim
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
import unicodedata
import re
from collections import Counter
register_heif_opener()

class GeoExtractor:
    def __init__(self, agent):
        self.agent_name = agent if agent else 'geoimage_extractor'

    def __remove_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = text.replace('Đ', 'D').replace('đ', 'd')
        return text

    def __get_exif_data(self, image_path):
        """Extract EXIF data from an image using Pillow."""
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return {}
        return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    
    def __get_heif_exif_data(image):
        """Extract EXIF data from an image."""
        exif_data = {}
        info = image.getexif()
        if info:
            for tag_id, value in info.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        return exif_data

    def __convert_to_degrees(self, value):
        """Convert GPS coordinates stored in EXIF to degrees."""
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    def __extract_coordinates(self, exif_data, filetype = 'non-heif'):
        """Extract GPS coordinates from EXIF data."""
        if filetype=='heif':
            gps_info = exif_data
        else:
            if 'GPSInfo' in exif_data:
                gps_info = exif_data['GPSInfo']
        gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
        # print(gps_data)
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat_values = gps_data['GPSLatitude']
            lon_values = gps_data['GPSLongitude']
            lat_ref = gps_data['GPSLatitudeRef']
            lon_ref = gps_data['GPSLongitudeRef']
            
            latitude = self.__convert_to_degrees(lat_values)
            longitude = self.__convert_to_degrees(lon_values)
            
            if lat_ref != 'N':
                latitude = -latitude
            if lon_ref != 'E':
                longitude = -longitude
            
            return latitude, longitude
        return None

    def __extract_address(self, latitude, longitude):
        """Convert coordinates to an address using geopy."""
        try:
            geolocator = Nominatim(user_agent=self.agent_name)
            location = geolocator.reverse((latitude, longitude), exactly_one=True, language='en')
            address = self.__remove_accents(location.address)
            return address
        except:
            return "UnknownAddress"

    def get_address(self, image_path):
        """Extract address from images (except HEIC/HEIF)"""
        if image_path.lower().endswith(('.heic', '.heif')):
            """Extract address from an HEIC image"""
            image = Image.open(image_path)
            image.verify()
            gpsexif = image.getexif().get_ifd(0x8825)
            location = self.__extract_coordinates(exif_data = gpsexif, filetype = 'heif')
        else:
            exif_data = self.__get_exif_data(image_path)
            location = self.__extract_coordinates(exif_data)
        if location:
            address = self.__extract_address(*location)
        else:
            address = "UnknownAddress"
        # print(f"Image: {image_path}")
        # print(f"Coordinates: {location}")
        # print(f"Address: {address}\n")
        return address
    
    def get_coordinates(self, image_path):
        """Process all images in a given directory."""
        exif_data = self.__get_exif_data(image_path)
        location = self.__extract_coordinates(exif_data)
        # print(f"Image: {image_path}")
        # print(f"Coordinates: {location}")
        return location

class SearchBM25:
    def __init__(self, embedding_dir="embed_store", searchfor='geo'):
        self.embedding_dir = embedding_dir
        self.searchfor = searchfor
        self.path = os.path.join(self.embedding_dir,f"{self.searchfor}_metadata.npy")
        # BM25 Parameters
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq = None
        self.N = None
        self.avgdl = None

    def __tokenize(self, query):
        return [tokens.strip().lower() for tokens in re.split(r",|:|'|\.|-|\s", query.split(sep="Location:")[-1]) if (tokens and len(tokens)>1)]
    
    def __compute_doc_freq(self, docs):
        df = {}
        for doc in docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
        return df
    
    def __compute_idf(self, term, doc_freq, N):
        df = doc_freq.get(term, 0)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def __bm25_score(self, query, doc, doc_index):
        score = 0.0
        doc_len = len(doc)
        term_freq = Counter(doc)
        for term in query:
            if term not in term_freq:
                continue
            idf = self.__compute_idf(term, self.doc_freq, self.N)
            freq = term_freq[term]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def search(self, query):
        tokenized_query = [token for token in self.__tokenize(query) if len(token)>2]
        book_dict = np.load(self.path, allow_pickle=True).item()
        book_items = list(book_dict.items()) 
        book_tokenized = [self.__tokenize(value) for _, value in book_items]
        self.N = len(book_tokenized)
        self.avgdl = sum(len(doc) for doc in book_tokenized) / self.N
        self.doc_freq = self.__compute_doc_freq(book_tokenized)

        scores = []
        for i, doc in enumerate(book_tokenized):
            score = self.__bm25_score(tokenized_query, doc, i)
            key, val = book_items[i]
            scores.append((score, key, val))
        ranked = sorted(scores, reverse=True)
        # print("Query:", query)
        # print("Ranked Documents:")
        # for score, index, address in ranked:
        #     print(f"Score: {score:.9f} - {index} - {address}")
        results = [key for score, key, val in ranked if score>0]
        return results

# if __name__=='__main__':
#     bm25geo = SearchBM25(searchfor='geo')
#     query = "Photos of statues in Vietnam"
#     filtered_indices = bm25geo.search(query)
#     del bm25geo
#     print(filtered_indices)


#     register_heif_opener()
#     geo = GeoExtractor(agent='my_agent')
#     image_path = 'ImageSamples/IMG_2129.HEIC'
#     addr = geo.get_address(image_path)

#     image_path = 'ImageSamples/IMG_6101.JPG'
#     addr = geo.get_address(image_path)
#     del geo
