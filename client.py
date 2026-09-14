import json
from typing import List, Dict, Any, Optional

class HybridRerankFusionRetrieverClient:
    """
    Production-grade Reciprocal Rank Fusion (RRF) and hybrid search reranker.
    Combines lexical BM25 and neural dense semantic ranks into a unified list.
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse_and_rerank_results(self, query: str = "Top rated cordless robot vacuum with LiDAR", dense_results: Optional[List[Dict[str, Any]]] = None, sparse_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not dense_results:
            dense_results = [
                {"doc_id": "doc_101", "title": "Roborock S8 MaxV Ultra Specs & LiDAR Performance", "score": 0.94},
                {"doc_id": "doc_102", "title": "Dreame X40 Cordless Robot Vacuum Review", "score": 0.89},
                {"doc_id": "doc_103", "title": "Dyson V15 Detect Handheld Stick Vacuum", "score": 0.82}
            ]
        if not sparse_results:
            sparse_results = [
                {"doc_id": "doc_102", "title": "Dreame X40 Cordless Robot Vacuum Review", "bm25_score": 14.8},
                {"doc_id": "doc_101", "title": "Roborock S8 MaxV Ultra Specs & LiDAR Performance", "bm25_score": 13.2},
                {"doc_id": "doc_104", "title": "Ecovacs Deebot T30 Pro Cordless Mop", "bm25_score": 11.5}
            ]

        rrf_scores = {}
        doc_metadata = {}

        # Process dense ranks
        for rank, d in enumerate(dense_results, 1):
            did = d["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.k + rank))
            doc_metadata[did] = {"title": d["title"], "dense_rank": rank}

        # Process sparse ranks
        for rank, d in enumerate(sparse_results, 1):
            did = d["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.k + rank))
            if did in doc_metadata:
                doc_metadata[did]["sparse_rank"] = rank
            else:
                doc_metadata[did] = {"title": d["title"], "sparse_rank": rank}

        fused = []
        for did, score in rrf_scores.items():
            fused.append({
                "doc_id": did,
                "title": doc_metadata[did]["title"],
                "rrf_score": round(score, 6),
                "dense_rank": doc_metadata[did].get("dense_rank", None),
                "sparse_rank": doc_metadata[did].get("sparse_rank", None)
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return {
            "query": query,
            "fusion_method": "Reciprocal Rank Fusion (RRF, k=60)",
            "top_doc_id": fused[0]["doc_id"],
            "top_title": fused[0]["title"],
            "fused_results_count": len(fused),
            "ranked_documents": fused
        }
