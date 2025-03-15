# Image Search with ChromaDB and BLIP

This project is a web application that allows users to search for images using text or audio queries. The application uses ChromaDB for image indexing and retrieval, and the BLIP model for generating image descriptions. Users can input their queries via text or microphone, and the application will return the most relevant images along with audio descriptions.

## Features

- **Text Query**: Users can input a text query to search for images.
- **Audio Query**: Users can use their microphone to input an audio query, which will be transcribed to text and used for the search.
- **Image Descriptions**: The application generates audio descriptions for each image result using the BLIP model.
- **Real-time Audio Streaming**: Uses `streamlit-webrtc` for real-time audio streaming from the microphone.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Malalane/ImageQuery.git
    cd ImageQuery
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create environment variables in the root directory using the following export on the terminal:
    
    ```sh
    export vectordb=your_vectordatabase_name
    ```

## Downloading Images

To download images for the dataset, follow these steps:

### Option 1: Download from Kaggle Website

1. **Go to the Kaggle competition page**: [Detect AI vs Human Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images)

2. **Download the dataset**: Follow the instructions on the Kaggle page to download the dataset to your local machine.

3. **Extract the dataset**: Extract the downloaded dataset to a directory of your choice.

### Option 2: Download using the Kaggle API

1. **Install the Kaggle API**:
    ```sh
    pip install kaggle
    ```

2. **Set up Kaggle API credentials**:
    - Go to your Kaggle account and create a new API token.
    - Download the `kaggle.json` file and place it in the `~/.kaggle/` directory (create the directory if it doesn't exist).

3. **Download the dataset using the Kaggle API**:
    ```sh
    kaggle competitions download -c detect-ai-vs-human-generated-images
    ```

4. **Extract the dataset**:
    ```sh
    unzip detect-ai-vs-human-generated-images.zip -d /path/to/dataset_folder
    ```

## Usage

### Creating a Vector Database

To create a vector database from the downloaded dataset of images, use the `create_vectordb.py` script. This script takes two arguments: the path to the dataset folder and the name of the vector database.

1. **Run the script**:
    ```sh
    python create_vectordb.py /path/to/dataset_folder my_vectordb_name
    ```

    Replace `/path/to/dataset_folder` with the actual path to your dataset folder and `my_vectordb_name` with the desired name for your ChromaDB vector database.

### Running the Streamlit Application

1. **Run the Streamlit application**:
    ```sh
    streamlit run streamlit_app.py
    ```

2. **Open your web browser** and go to `http://localhost:8501`.

3. **Enter your query**:
    - **Text Query**: Enter your query text in the input box.
    - **Audio Query**: Click on the microphone button to start recording your query.

4. **View Results**: The application will display the most relevant images along with audio descriptions.

5. **Install Rosetta on macOS**:
    If you are using macOS, you will need to install Rosetta to get audio to work on Streamlit:
    ```sh
    softwareupdate --install-rosetta
    ```

## File Structure

```
project/
│
├── vectordatabase/
│
├── create_vectordb.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Dependencies

- `streamlit`
- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `chromadb`
- `SpeechRecognition`
- `gtts`
- `streamlit-webrtc`
- `transformers`
- `keybert`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [ChromaDB](https://chromadb.com/)
- [Google Text-to-Speech](https://pypi.org/project/gTTS/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

