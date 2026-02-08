# celebrity lookalike

finds which celebrity you look like

## how to run

```bash
pip install -r requirements.txt
python app.py
```

then go to http://localhost:5000

## hosting

use render.com or railway.app (github pages doesnt work for flask apps)

on render:
- new web service
- connect github repo
- build: `pip install -r requirements.txt`
- start: `gunicorn app:app --bind 0.0.0.0:$PORT`

## stuff

- put celeb photos in the `celebs` folder
- uses face recognition to match
- webcam or upload photo

