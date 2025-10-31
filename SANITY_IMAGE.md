# Image preview sanity walkthrough

Use this checklist to confirm the built-in image sample plan works end-to-end and renders the thumbnail table correctly.

## 1. Start the FastAPI backend

Open a terminal, install backend dependencies if needed, then launch the service:

```bash
./dev-run.sh
```

Keep the process running and wait for Uvicorn to print that it is serving `http://127.0.0.1:8000`.

## 2. Launch the Streamlit UI

In a second terminal (same virtual environment), install the UI requirements and start Streamlit:

```bash
python -m pip install -r ui/requirements.txt
streamlit run ui/streamlit_app.py
```

When Streamlit shows the local URL (usually `http://localhost:8501`), open it in your browser.

## 3. Switch to Images mode

Inside the UI sidebar, choose **Images** under **Dataset type**. The sidebar now shows upload controls plus the bundled helpers.

Confirm the helper caption lists three bundled assets located at `artifacts/bundled_images/`. The UI writes
tiny placeholder PPM images into that directory automatically the first time you open Images mode, so no manual
setup is required.

## 4. Load the bundled sample plan

Click **Load sample plan (images)**. The app should immediately:

- Populate the operations list with two steps: `img_caption` (instructions `用一句中文描述图片内容`,
  max tokens 80) and `img_resize` (width/height `384`, keep ratio enabled).
- Select the bundled images (`forest.ppm`, `ocean.ppm`, `sunrise.ppm`) and show their thumbnails in the main panel.
- Adjust the **Preview sample size** slider to 3.

If any of those conditions fail, make sure the Streamlit process has permission to write to
`artifacts/bundled_images/` so it can regenerate the placeholder PPM files.

## 5. Run Preview

Press **Preview**. The backend receives a `/preview` request that references only the three bundled files (limited by the slider).

In the **Preview output** section, confirm the rendered table has:

- A **Thumb** column that displays 64px thumbnails of each bundled image.
- A **Filename** column listing `forest.ppm`, `ocean.ppm`, and `sunrise.ppm`.
- A **Caption** column filled by the `img_caption` step (Chinese one-sentence descriptions).
- A **Resized path** column pointing to temporary files under `artifacts/tmp/.../img_resize/`.

The row count should match the slider value (3 by default).

## 6. (Optional) Adjust the sample size

Move the **Preview sample size** slider to 1 or 2 and click **Preview** again. The table should refresh to show only that many rows, confirming the request payload respects the slider limit.

With these checks complete, the bundled image workflow is ready for manual QA runs.
