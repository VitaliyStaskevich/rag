# RAG (Retrieval-Augmented Generation) system

This project is a customizable RAG creation and inference application, designed for easy and clean chunking of .pdf files with dense vectors.

The current as-is code is used for making the RAG based on a Civil Law Code.

# Chunking Features
* Instead of fixed-length blocks, the system splits text by legal articles to preserve legal logic.
* The actual chunking logic is mostly contained to `article_pattern` regex, so it can easily be customized to fit other document styles.

# Inference

The interface for using the RAG is created using `Streamlit`.

# Inference features
* The retrival part is highly custimizable: you can choose the `top_k` and `n_neigbors` parameters.
* Neibor retrival (context windows) is especially important if the document that is being chunked is context-sensetive.
* Chat history is persistent - everyting is saved into the json format in `chat_sessions` folder.
* LaTeX support for rendering formulas.

# Prerequisites

Before running the application, ensure you have:
1. Mistral API Key: Get one at [Mistral AI](https://console.mistral.ai)
2. Pinecone API Key: Create a free account at [Pinecone](https://www.pinecone.io)
3. The relevant docment you want to use (in the .pdf format in the same directory as scripts)

# Installation & Setup

1. Clone the repository:
```
git clone https://github.com/VitaliyStaskevich/rag.git
cd rag
```
2. Install requirements:
```
pip install -r requirements.txt
```
3. Fill in your api keys in both `chunking.py` and `streamlit.py` (optionally, in the `cleanup.py`) files:
```
MISTRAL_API_KEY = "your_mistral_key"
PINECONE_API_KEY = "your_pinecone_key"
```
4. Configure the `chunking.py` script:
```
INDEX_NAME = "" - the name of the index in the Pinecone database
PDF_PATH = ""  - name of the file, including the .pdf extension
START_PAGE = - starting page of the relevant info in the file (to skip intoductions, table of contents, etc.)
article_pattern = '' - regex pointing out how to seperate the chunks 
```
This script uses the `BAAI/bge-m3` model for vector embeddings, which generates 1024-dimensional vectors. However, it may be too hardware-intensive for some devices. In that case you should either uncomment the
```
#model = SentenceTransformer('all-MiniLM-L6-v2')
```
line and change the index dimensions to 384, which will reduce the computational complexity of the rag (as well as the retrival accuracy).

Alternatively, any model from [Hugging face](https://huggingface.co) can be used, just make sure the dimensions of the vectors are correct.
## Usage

Run
```
py chunking.py
```
to generate an index.
After that you may optionally use the `cleanup.py` script to filter out the junk indecies, by filling out the unwanted words into the `trash_patterns` array and running the script
```
py cleanup.py
```
After the index is built, you can use the interface to fetch info from the document.
Firstly, configure the `streamlit.py` file:
```
PINECONE_INDEX_NAME = "" - the name MUST match the one in the chunking script
MODEL_MISTRAL = "mistral-small-latest" - recomended model, but you may opt in to use any one you like
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3' - the embedding model MUST match the one in the chunking script, else the results will be wildly inaccurate
top_k_param = 10 - relevant chuncks that will be fetched for each query. Recomended to scale it up and down in accordance to the size of the chunks
N_NEIGHBORS = 0 - if the info is not context-sensetive, leave it at 0
SYSTEM_PROMPT = "" - system prompt for the LLM portion of the RAG. Highly advised to be as specific as possible for the best results.
```
After the setup, run
```
streamlit run streamlit.py
```
And you can immediatly start getting info about the document.

# 🛠 Technology Stack
* Pinecone vector database,
* Sentence Transformers (BAAI/bge-m3) for vector embeddings,
* pypdf for PDF processing,
* Mistral AI (mistral-small-latest) as an LLM for generation,
* streamlit for the frontend
  
