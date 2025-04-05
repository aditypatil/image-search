# Image Search with ChromaDB and BLIP

This project 

## Features


## Installation

1. **Clone the repository**:


2. **Create a virtual environment**:


3. **Install the dependencies**:


4. **Set up environment variables**:


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

