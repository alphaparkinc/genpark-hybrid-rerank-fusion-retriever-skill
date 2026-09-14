import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from client import HybridRerankFusionRetrieverClient

def main():
    client = HybridRerankFusionRetrieverClient()
    res = client.fuse_and_rerank_results()
    print("=== Hybrid Rerank Fusion Retriever Output ===")
    print(f"Query: {res['query']}")
    print(f"Top Pick: {res['top_title']} (RRF: {res['ranked_documents'][0]['rrf_score']})")
    print(f"Total Fused Documents: {res['fused_results_count']}")
    print("\nFused Ranking List:")
    for d in res['ranked_documents']:
        print(f"  - [{d['rrf_score']:.6f}] {d['doc_id']}: {d['title']} (Dense: {d['dense_rank']}, Sparse: {d['sparse_rank']})")

if __name__ == '__main__':
    main()
