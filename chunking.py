import os
import re
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ---------------- CONFIG ----------------
PINECONE_API_KEY = ""
INDEX_NAME = ""
PDF_PATH = "" 
START_PAGE = 37 
# ----------------------------------------

def load_pdf_from_page(path, start_page_num):
    print(f"Reading file {path} starting from page {start_page_num}...")
    reader = PdfReader(path)
    text = ""
    
    # the pages in the pypdf are 0-indexed
    start_index = start_page_num - 1
    total_pages = len(reader.pages)
    
    for i in range(start_index, total_pages):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += f"\n--- PAGE {i+1} ---\n" 
            text += page_text
            
    return text

def chunk_by_articles(text):
    article_pattern = r'(?m)^\s*(?=Статья\s+\d+)'
    raw_chunks = re.split(article_pattern, text)
    
    print(f"Found {len(raw_chunks)} fragments in total")

    body_chunks = []
    max_article_num = 0
    toc_passed = False

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        match = re.search(r'Статья\s+(\d+)', chunk)
        if match:
            current_num = int(match.group(1))
            
            if not toc_passed:
                if current_num < max_article_num:
                    toc_passed = True
                    body_chunks.append(chunk)
                    print(f"Text starta at №{current_num}")
                else:
                    max_article_num = current_num
                    continue
            else:
                body_chunks.append(chunk)
        else:
            if toc_passed and body_chunks:
                body_chunks[-1] += "\n\n" + chunk

    return body_chunks

def main():
    full_text = load_pdf_from_page(PDF_PATH, START_PAGE)
    
    print("Chunkinh and filtering out the contents...")
    chunks = chunk_by_articles(full_text)
    print(f"{len(chunks)} chunks are ready for loading.")
    procceed = input("Continue? y/n")
    if procceed == 'n':
        print('Stopping')
        return 0
    else:
        print("Continuing...")
    if not chunks:
        print("No chunks found, check the file structure.")
        return

   
    #print("Generating embeddings (all-MiniLM-L6-v2)...")
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings (BAAI/bge-m3)...")
    model = SentenceTransformer('BAAI/bge-m3')
    embeddings = model.encode(chunks, show_progress_bar=True)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        print(f"Creating index {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            #dimension=384,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(INDEX_NAME)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        i_end = min(i + batch_size, len(chunks))
        
        ids = [f"art_{x}" for x in range(i, i_end)]
        metadatas = [{"text": chunks[x]} for x in range(i, i_end)]
        vectors = embeddings[i:i_end]

        records = zip(ids, vectors, metadatas)
        index.upsert(vectors=list(records))
        print(f"Loaded {i_end}/{len(chunks)} chunks.")

    print("Done, the index is created/updated")

if __name__ == "__main__":
    main()