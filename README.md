# celebrity lookalike

finds which celebrity you look like

## how to run

```bash
pip install -r requirements.txt
python app.py
```

then go to http://localhost:5000

## hosting

### Hugging Face Spaces (Recommended Free Option)
Hugging Face Spaces is excellent for hosting ML/CV apps like this for free.

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space).
2. Select **Streamlit** (for the Streamlit version) or **Docker** (for the Flask version).
3. Upload/Push your code.
4. Go to **Settings** -> **Variables and Secrets**.
5. Add `TMDB_API_KEY` as a **Secret**.

### Render.com
1. Create a **New Web Service**.
2. Connect your GitHub repo.
3. Build Command: `chmod +x build.sh && ./build.sh`
4. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Go to **Environment** and add `TMDB_API_KEY`.
6. Note: The free tier has 512MB RAM, which might be tight for `face_recognition`.

### Streamlit Cloud
If you prefer the Streamlit version (`streamlit_app.py`):
1. Deploy via [streamlit.io/cloud](https://streamlit.io/cloud).
2. Add your `TMDB_API_KEY` in the **Secrets** section of the app settings.

### Koyeb
Koyeb has a solid free "Nano" instance that works well with Docker and Python.
1. Create a new service on [koyeb.com](https://www.koyeb.com/).
2. Connect your repo (the `Procfile` is already included to handle the start command).
3. Set the `TMDB_API_KEY` environment variable in the Koyeb settings.

### What about Cloudflare?
Cloudflare Pages and Workers are great for static sites and small scripts, but they **cannot** run this app. This is because:
- **Heavy Dependencies**: Libraries like `face_recognition` and `mediapipe` are too large for Cloudflare's memory limits.
- **Python Support**: Cloudflare Workers' Python support is limited and doesn't easily support binary libraries like `opencv` or `dlib`.

## environment variables
This app requires the following environment variables:
- `TMDB_API_KEY`: Your The Movie Database API key (optional but recommended for better image results).

## stuff

- put celeb photos in the `celebs` folder
- uses face recognition to match
- webcam or upload photo
