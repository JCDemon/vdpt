# Image workflow sanity walkthrough

Follow this guided flow to manually validate that the multimodal pipeline can ingest raw images, generate previews, and produce saved artifacts.

## 1. Start the backend API

Run the FastAPI backend in a dedicated terminal so the UI can talk to it:

```bash
./dev-run.sh
```

Leave the server running; you should see Uvicorn logs confirming that the app is serving on `http://127.0.0.1:8000`.

> _Screenshot placeholder – backend terminal output_

## 2. Launch the Streamlit UI

In a separate terminal (with the same virtual environment activated), launch the UI:

```bash
python -m pip install -r ui/requirements.txt
streamlit run ui/streamlit_app.py
```

Wait for Streamlit to report the local URL (typically `http://localhost:8501`), then open it in your browser.

> _Screenshot placeholder – Streamlit home screen_

## 3. Prepare sample images (optional helper)

The repository ships with a few demo assets under `samples/images/`. To quickly reuse them across sessions, you can seed the current upload directory with:

```bash
./dev-seed-images.sh <session-id>
```

Replace `<session-id>` with the identifier shown in the UI sidebar (e.g. `20240515-153000`). The script copies `.png` and `.jpg` files into `artifacts/uploads/<session-id>/images/` so they appear in the file picker without manually uploading each time.

## 4. Upload three images

If you did not run the helper script, use the **Upload files** widget in the Streamlit sidebar to add any three images (PNG or JPG). The uploaded files will appear in the session's image library.

> _Screenshot placeholder – image uploads in sidebar_

## 5. Build a plan with `img_caption` and `img_resize`

1. In the **Plan editor** panel, add an `img_caption` operation. Configure it to read from the uploaded images collection.
2. Add a second step `img_resize` that depends on the captions and resizes the images to a smaller resolution (e.g. `512x512`).
3. Save the plan.

Ensure the preview count at the top of the panel is set to **3** so every uploaded image is sampled during preview.

> _Screenshot placeholder – plan editor with two steps_

## 6. Preview the plan

Click **Preview**. The UI should execute the plan with `preview=3` and display both the generated captions and resized image thumbnails in the results pane.

> _Screenshot placeholder – preview results_

## 7. Execute the full run

Once the preview looks correct, click **Execute**. The UI will run the entire workflow and report the run identifier plus the artifacts directory path (e.g. `artifacts/runs/<run-id>`).

> _Screenshot placeholder – execution confirmation_

## 8. Inspect generated artifacts

Open the reported directory to review the outputs. You should find:

- Generated captions (JSON or text depending on provider configuration).
- Resized images under an `img_resize`-specific folder.
- Provenance metadata for the run.

You can list the directory contents from the terminal, substituting the actual run identifier:

```bash
ls -R artifacts/runs/<run-id>
```

> _Screenshot placeholder – artifacts directory listing_

If every step above completes successfully, the image pipeline is functioning end-to-end.
