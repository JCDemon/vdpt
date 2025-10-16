# Sanity checklist

Use this checklist to verify the MVP locally before opening a pull request.

1. **Install dependencies**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r backend/requirements.txt pytest
   ```
2. **Run the automated tests**
   ```bash
   pytest
   ```
3. **Start the development server**
   ```bash
   ./dev-run.sh
   ```
4. **Verify endpoints manually**
   - `GET /health` should return `{ "status": "ok" }`.
   - `POST /todos` with a payload such as `{ "title": "Manual test" }` should return a created todo with an `id` and `completed: false`.
   - `POST /todos/{id}/complete` should mark the todo as completed.
5. **Stop the server** with `Ctrl+C` once checks complete.
