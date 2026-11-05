import json
import time
from pathlib import Path
from search import search_v0
from dotenv import load_dotenv

load_dotenv()

def evaluate_v0():
    golden_path = Path("golden.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    results_log = []
    hits = 0
    mrr_sum = 0.0

    for q in golden:
        start_time = time.time()
        
        retrieved = search_v0(q["question"], n=8)
        latency = time.time() - start_time
        
        retrieved_docs = [c["doc_id"] for c in retrieved]
        target_files = q.get("target_files", [])
        
        hit = False
        hit_rank = None
        reciprocal_rank = 0.0
        
        if target_files:
            for rank, doc in enumerate(retrieved_docs, start=1):
                if any(target in doc or doc in target for target in target_files):
                    hit = True
                    hit_rank = rank
                    reciprocal_rank = 1.0 / rank
                    break
        
        if hit:
            hits += 1
        mrr_sum += reciprocal_rank

        results_log.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "hit": hit,
            "hit_rank": hit_rank,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "latency_sec": round(latency, 3),
            "retrieved_docs": retrieved_docs
        })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "before.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    total_q = len(golden)
    valid_q = len([q for q in golden if q.get("target_files")])
    mrr = round(mrr_sum / valid_q, 4) if valid_q > 0 else 0.0

    print(f"Evaluation complete for v0.")
    print(f"Total Hits@8: {hits}/{total_q}")
    print(f"MRR: {mrr}")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    evaluate_v0()