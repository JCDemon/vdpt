## Summary
- introduce a dataset loader registry with scan/preview support in the backend
- add COCO 2017, Cityscapes fine, and HuggingFace image dataset loader implementations
- expose `/datasets/list` and `/datasets/preview` APIs with simple caching to power UI previews
- add a Streamlit dataset picker sidebar that renders a 12-image preview grid with metadata

## Testing
- `python -m compileall backend/app`
- `python -m compileall ui`

## Manual verification
```bash
uvicorn backend.app.main:app --reload
```

```bash
streamlit run ui/streamlit_app.py
```

1. In the Streamlit sidebar, set the backend URL to `http://127.0.0.1:8000`.
2. Open the **Datasets** panel, pick the **COCO 2017** loader, and point it at `artifacts/sample_data/coco2017_tiny`.
3. Click **Preview dataset** to render the 12-sample thumbnail grid.
4. Repeat with the **HuggingFace datasets** loader using `beans` as the dataset name.

## Notes
- Cityscapes access requires registration and acceptance of the license terms: https://www.cityscapes-dataset.com/
- HuggingFace datasets require the `datasets` Python package and internet connectivity for first-time downloads.
