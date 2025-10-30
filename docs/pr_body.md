## Summary
- add OpenCLIP text and image embedding operations with disk caching for preview runs
- reduce embeddings with UMAP, cluster with HDBSCAN, and expose new artifacts
- surface a Streamlit cluster view with interactive point inspection

## Testing
- `python -m compileall backend/app`
- `python -m compileall ui`

## Manual verification
```bash
curl -s http://127.0.0.1:8000/preview -H 'Content-Type: application/json' -d '{
  "dataset":{"type":"csv","path":"artifacts/uploads/sample_news.csv"},
  "preview_sample_size": 50,
  "operations":[
    {"kind":"field","field":"text"},
    {"kind":"embed_text","field":"text"},
    {"kind":"umap","source":"embedding"},
    {"kind":"hdbscan","source":"umap"}
  ]}' | jq '.artifacts | keys'
```

```bash
curl -s http://127.0.0.1:8000/preview -H 'Content-Type: application/json' -d '{
  "dataset":{"type":"images","path":"artifacts/bundled_images"},
  "preview_sample_size": 20,
  "operations":[
    {"kind":"embed_image"},
    {"kind":"umap","source":"embedding"},
    {"kind":"hdbscan","source":"umap"}
  ]}' | jq '.artifacts | keys'
```
