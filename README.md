# Image Search System

## Overview

This project provides a robust system for image search and metadata extraction, leveraging advanced machine learning models and indexing techniques. It integrates face detection, CLIP-based image search, and geolocation metadata extraction to enable efficient querying of images based on textual descriptions, face labels, and geographic information. The system is designed for scalability and modularity, making it suitable for applications requiring sophisticated image retrieval and analysis.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [References](#references)
- ~[License](#license)~

## Features

- **Face Detection and Clustering**: Utilizes InsightFace for face detection and feature extraction, with Agglomerative Clustering to assign labels to detected faces.
- **CLIP-based Image Search**: Employs the CLIP model (Contrastive Language–Image Pre-training) to generate image embeddings and perform text-based image retrieval using FAISS for efficient similarity search.
- **Geolocation Metadata Extraction**: Extracts GPS coordinates and addresses from image EXIF data using Pillow and Nominatim, supporting both standard image formats and HEIC/HEIF.
- **BM25 Search**: Implements BM25Okapi for text-based search on face labels and geolocation metadata.
- **Combined Search Strategy**: Integrates face detection, geolocation, and CLIP-based search to refine results based on query context.
- **Data Ingestion Pipeline**: Automates the generation and storage of embeddings and metadata for a given image directory.

## Architecture

The system is modular, with distinct components for ingestion, search, and metadata extraction:

1. **Ingestion (`ingestion.py`)**: Orchestrates the generation of image paths, CLIP embeddings, face data, and geolocation metadata. Stores results in a designated embedding directory.
2. **Face Detection (`face_detection.py`)**: Detects faces using InsightFace, extracts embeddings, and clusters them to assign labels. Supports BM25-based search on face labels.
3. **CLIP Search (`clip_search.py`)**: Generates image embeddings using CLIP and supports FAISS-based similarity search for text queries, with optional subset filtering.
4. **Geolocation Metadata (`geo_metadata.py`)**: Extracts GPS coordinates and addresses from images, with BM25-based search on address data.
5. **Search Strategy (`retrieval.py`)**: Combines face detection, geolocation, and CLIP search results using a conditional strategy (intersection or union of indices) to optimize query results.

## Installation

### Prerequisites

- Python 3.8+
- A CUDA-capable GPU (optional, for faster CLIP inference)
- Virtual environment (recommended)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aditypatil/image-search.git
   cd image-search-system
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Embedding Directory**:
   Create a directory (e.g., `embed_store`) to store embeddings and metadata:
   ```bash
   mkdir embed_store
   ```

5. **Prepare the Image Directory**:
   Place images in a directory (e.g., `ImageSamples`) with supported formats (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`).

## Usage

### Data Ingestion

Run the ingestion pipeline to process images and generate embeddings/metadata:

```bash
python ingestion.py
```

This will:
- Generate image paths and store them in `embed_store/img_path_index.pkl`.
- Create CLIP embeddings (`img_embeddings.bin`, `img_filenames.npy`).
- Generate face data (`face_data.pkl`, `img_path_index_for_face.pkl`).
- Extract geolocation metadata (`geo_metadata.npy`).

### Search

Perform a combined search using face labels, geolocation, and CLIP embeddings:

```bash
python retrieval.py
```

Example query:
```python
srch = Search()
img_indices = srch.strategy1(query="aditya on beaches of goa")
print(img_indices)
```

This query searches for images of a person labeled "Aditya" in locations associated with "Goa beaches," refined by CLIP-based text similarity.

### Adding Face Labels

~To assign meaningful names to face clusters, modify `face_detection.py`'s `add_dummy_faces` function and run:~

```bash
python face_detection.py
```

Example mapping:
```python
name_labels = {0: "Aditya", 1: "Pratik", 2: "Dogemaster", 3: "NeoBoomer", 4: "KittyCat"}
```

## File Structure

```plaintext
image-search-system/
├── embed_store/                  # Directory for storing embeddings and metadata
├── ImageSamples/                 # Directory for input images
├── models/
│   ├── __init__.py
│   ├── clip_search.py            # CLIP embedding generation and search
│   ├── face_detection.py         # Face detection and clustering
│   └── geo_metadata.py           # Geolocation metadata extraction
├── retrieval.py                  # Combined search strategy
├── ingestion.py                  # Data ingestion pipeline
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

## Dependencies

- **FAISS**: For efficient similarity search of CLIP embeddings.
- **InsightFace**: For face detection and feature extraction.
- **CLIP (Transformers)**: For text-image embedding generation.
- **Pillow**: For image processing and EXIF data extraction.
- **Geopy**: For reverse geocoding of GPS coordinates.
- **scikit-learn**: For clustering face embeddings.
- **rank-bm25**: For text-based search on face labels and addresses.
- **tqdm**: For progress bars during processing.
- **NumPy, Pandas, Matplotlib, OpenCV**: For data manipulation and visualization.
- **Torch**: For CLIP model inference.
- **Pillow-HEIF**: For HEIC/HEIF image support.

## References

- Szegedy, C., et al. (2015). "Going Deeper with Convolutions." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [InsightFace model inspiration]
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *International Conference on Machine Learning (ICML)*. [CLIP model]
- Jones, K. S., et al. (2000). "A Probabilistic Model of Information Retrieval: Development of the BM25 Model." *Journal of Documentation*. [BM25 algorithm]
- FAISS Documentation: https://github.com/facebookresearch/faiss
- InsightFace Documentation: https://github.com/deepinsight/insightface
- Geopy Documentation: https://geopy.readthedocs.io/
- Transformers (Hugging Face) Documentation: https://huggingface.co/docs/transformers/

## License

~This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.~
