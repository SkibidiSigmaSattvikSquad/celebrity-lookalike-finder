# celebrity lookalike

finds which celebrity you look like

## how to run

```bash
pip install -r requirements.txt
python app.py
```

then go to http://localhost:5000

## hosting

### Koyeb (Recommended)
This app has complex dependencies. Using the **Docker builder** is much more reliable than Buildpacks.

1. Create a new service on [koyeb.com](https://www.koyeb.com/).
2. Connect your repo.
3. In the **Builder** section, select **Docker** (it will use the `Dockerfile` in the repo).
4. In the **Environment** section, add your `TMDB_API_KEY`.
5. In the **Exposed Port** section, set it to **8000**.
6. **Instance**: If it crashes, you may need a larger instance than "Nano" (512MB) for face recognition.

### Hugging Face Spaces
1. Create a new Space on [huggingface.co](https://huggingface.co/new-space).
2. Select **Docker**.
3. Push your code and add `TMDB_API_KEY` as a **Secret**.

### Render.com
1. Create a **New Web Service**.
2. Connect your GitHub repo.
3. Build Command: `chmod +x build.sh && ./build.sh`
4. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

## environment variables
- `TMDB_API_KEY`: Your The Movie Database API key.
