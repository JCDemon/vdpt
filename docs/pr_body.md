## Summary
- integrate SAM and CLIPSeg segmentation operations for generic and promptable mask generation ([Segment Anything](https://github.com/facebookresearch/segment-anything), [CLIPSeg](https://arxiv.org/abs/2112.10003))
- embed segmented masks with OpenCLIP, project with UMAP, and cluster with HDBSCAN while persisting mask-level artifacts
- add a Streamlit mask analytics view with colorized overlays, per-mask tables, and one-click PNG exports

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

```bash
curl -s http://127.0.0.1:8000/preview -H 'Content-Type: application/json' -d '{
  "dataset":{"type":"images","path":"artifacts/bundled_images","paths":["forest.png","ocean.png","sunrise.png"]},
  "preview_sample_size": 3,
  "operations":[
    {"kind":"sam_segment"},
    {"kind":"embed_masks"},
    {"kind":"umap","source":"mask_embedding"},
    {"kind":"hdbscan","source":"umap"}
  ]}' | jq '.artifacts | keys'
```
