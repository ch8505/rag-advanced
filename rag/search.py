import json
import numpy as np
import voyageai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()

CHUNKS_PATH = Path("corpus_chunks.json")
VECS_PATH = Path("v0_embeddings.npy")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

VECS = np.load(VECS_PATH)

def search_v0(query: str, n: int = 8) -> list[dict]:
    """חיפוש וקטורי בלבד (Top-N) עבור גרסה v0"""
    res = vo.embed([query], model="voyage-3-lite", input_type="query")
    qv = np.array(res.embeddings[0])
    
    scores = np.dot(VECS, qv)
    
    top_indices = np.argsort(-scores)[:n]
    
    results = []
    for idx in top_indices:
        chunk = CHUNKS[idx].copy()
        chunk["score"] = float(scores[idx])
        results.append(chunk)
        
    return results