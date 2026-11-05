import os
import re
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()

CHUNK_SIZE = 1000

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    text_for_index: str = ""

def resolve_code_imports(content: str, base_dir: Path) -> str:
    pattern = r"\{\*\s*\.\./\.\./docs_src/([^\s\*\}]+)[^\}]*\*\}"
    
    def replacer(match: re.Match) -> str:
        rel_path = match.group(1)
        code_file = base_dir / "docs_src" / rel_path
        if code_file.exists():
            code_content = code_file.read_text(encoding="utf-8")
            return f"\n```python\n{code_content}\n```\n"
        return match.group(0)

    return re.sub(pattern, replacer, content)

def chunk_fixed_size(doc_id: str, title: str, content: str, chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
    chunks = []
    for idx, i in enumerate(range(0, len(content), chunk_size)):
        sec_text = content[i:i + chunk_size].strip()
        if not sec_text:
            continue
            
        c_id = f"{doc_id}#chunk-{idx+1}"
        chunks.append(Chunk(
            chunk_id=c_id,
            doc_id=doc_id,
            doc_title=title,
            text=sec_text,
            text_for_index=sec_text
        ))
    return chunks

def load_corpus_v0(fastapi_dir: Path) -> List[Chunk]:
    docs_dir = fastapi_dir / "docs" / "en" / "docs"
    targets = [docs_dir / "tutorial", docs_dir / "advanced"]
    
    all_chunks = []
    
    for target in targets:
        for root, _, files in os.walk(target):
            for file in files:
                if file.endswith(".md"):
                    file_path = Path(root) / file
                    rel_doc_id = str(file_path.relative_to(fastapi_dir)).replace("\\", "/")
                    
                    content = file_path.read_text(encoding="utf-8")
                    resolved_content = resolve_code_imports(content, fastapi_dir)
                    title = file_path.stem.replace("-", " ").title()
                    
                    doc_chunks = chunk_fixed_size(rel_doc_id, title, resolved_content)
                    all_chunks.extend(doc_chunks)
                    
    return all_chunks

def ingest_v0():
    fastapi_path = Path("corpus/fastapi")
    print("Ingesting corpus for v0...")
    chunks = load_corpus_v0(fastapi_path)
    print(f"Total chunks created: {len(chunks)}")
    
    out_path = Path("corpus_chunks.json")
    out_path.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print("Generating embeddings via Voyage AI...")
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c.text for c in chunks[i:i + batch_size]]
        result = vo.embed(batch_texts, model="voyage-3-lite", input_type="document")
        all_embeddings.extend(result.embeddings)
        print(f"Embedded {len(all_embeddings)}/{len(chunks)}")
        
    np.save("v0_embeddings.npy", np.array(all_embeddings))
    print("Ingestion complete. Saved corpus_chunks.json and v0_embeddings.npy")

if __name__ == "__main__":
    ingest_v0()