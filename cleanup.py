import re
from pinecone import Pinecone

# ---------------- CONFIG ----------------
PINECONE_API_KEY = ""
INDEX_NAME = ""
ID_PREFIX = "art_" 
MAX_ARTICLES = 1500 
# ----------------------------------------

def is_trash_article(text):
    text_clean = text.strip()
    trash_patterns = [
        r"исключена", 
        r"утратила силу", 
        r"норма исключена"
    ]
    

    if len(text_clean) < 150:
        for pattern in trash_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
    return False

def cleanup_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    to_delete = []
    batch_size = 100
    
    print("Scanning index...")
    
    for i in range(0, MAX_ARTICLES, batch_size):
        current_ids = [f"{ID_PREFIX}{j}" for j in range(i, i + batch_size)]
        
        res = index.fetch(ids=current_ids)
        
        for vid, data in res['vectors'].items():
            metadata = data.get('metadata', {})
            text = metadata.get('text', '')
            
            if is_trash_article(text):
                print(f"Indecies to be deleted: [{vid}]: {text[:60]}...")
                to_delete.append(vid)
    
    if to_delete:
        print(f"Found {len(to_delete)} indecies to be deleted.")
        # Удаляем пачками по 100
        for i in range(0, len(to_delete), 100):
            chunk = to_delete[i : i + 100]
            index.delete(ids=chunk)
            print(f"Deleted {len(chunk)} vectord...")
        print("Done.")
    else:
        print("No junk found.")

if __name__ == "__main__":
    cleanup_pinecone()