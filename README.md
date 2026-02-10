# celebrity lookalike

finds which celebrity you look like (Flask Version)

## how to run locally

```bash
pip install -r requirements.txt
python app.py
```

## hosting (The "Easy" Way)

### ❌ GitHub Pages / Netlify / Vercel
These will **not** work for this app. They are for "static" sites (HTML/JS only). This app needs a **Python Server** to run the face-recognition AI.

### ✅ Hugging Face Spaces (Easiest for Flask)
This is effectively "GitHub Pages for AI apps." It's the only free place that reliably runs the heavy libraries this app uses.

1. **Push your code**: `git push`
2. Create a **[New Space](https://huggingface.co/new-space)**.
3. Select **Docker** as the SDK.
4. **Connect GitHub**: Select your `celebrity-lookalike-finder` repo.
5. **Add Secret**: In Settings, add `TMDB_API_KEY` so celebrity photos show up.
6. **Done**: It will build and give you a public URL.

## environment variables
- `TMDB_API_KEY`: Your The Movie Database API key.
