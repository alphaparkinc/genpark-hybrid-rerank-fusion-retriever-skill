import json, sys
from client import HybridRerankFusionRetrieverClient

def handle_mcp_request(payload):
    method = payload.get("method")
    req_id = payload.get("id", 1)
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "hybrid-rerank-fusion", "version": "1.0.0"}}}
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [{"name": "fuse_and_rerank_results", "description": "Fuses dense semantic vectors and sparse lexical BM25 ranks via Reciprocal Rank Fusion."}]}}
    elif method == "tools/call":
        client = HybridRerankFusionRetrieverClient()
        res = client.fuse_and_rerank_results()
        return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
    return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ACTIVE"}}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print(json.dumps(handle_mcp_request({"method": "tools/list"})))
    else:
        client = HybridRerankFusionRetrieverClient()
        print(json.dumps(client.fuse_and_rerank_results(), indent=2))
