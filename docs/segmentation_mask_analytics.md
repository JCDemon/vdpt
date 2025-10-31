# Segmentation Mask Analytics

## Overview

The mask analytics workflow adds end-to-end support for image segmentation, per-mask OpenCLIP embeddings, dimensionality reduction, and clustering. Two segmentation strategies are available:

* **Segment Anything (SAM)** via the `sam_segment` operation.
* **CLIPSeg** (text guided) via the `clipseg_segment` operation.

For each segmentation mask we persist a binary PNG mask, the associated bounding box and metadata, the normalized OpenCLIP embedding vector, 2D UMAP coordinates, and HDBSCAN cluster labels.

## Requirements & Caveats

* A ViT-B SAM checkpoint must be available on disk. Point `SAM_CHECKPOINT_PATH` to the checkpoint file or provide `checkpoint_path` in the `sam_segment` parameters.
* CLIPSeg weights are downloaded from Hugging Face on first use. Ensure the environment has network access or pre-populate the cache.
* All models default to CPU execution when CUDA is unavailable. Expect slower inference on larger batches.
* The UI mask overlay loads mask PNGs directly from the artifacts directory. Keep preview runs to a manageable size to avoid excessive filesystem writes.

## Operations pipeline

A typical analytics pipeline looks like:

1. `sam_segment` **or** `clipseg_segment`
2. `embed_masks` (produces `mask_embedding` vectors and writes `.npy` files)
3. `umap` with `source="mask_embedding"` and `output_field="mask_umap"`
4. `hdbscan` with `source="mask_umap"` and `output_field="mask_cluster"`

All mask-related artifacts are stored under the run directory (e.g. `artifacts/run-*/masks/`).

## UI usage

The Streamlit app now exposes a **Mask analytics** panel when working with image datasets. The panel allows you to:

* Select the segmentation backend (SAM or CLIPSeg) and provide an optional prompt for CLIPSeg.
* Trigger the analytics pipeline described above in a single click.
* Visualize mask embeddings and clusters on a 2D scatter plot.
* Inspect per-mask metadata, view color-coded overlays, and download all mask PNGs for each image as a ZIP archive.

Reloading or changing the dataset clears the cached mask analytics results.
