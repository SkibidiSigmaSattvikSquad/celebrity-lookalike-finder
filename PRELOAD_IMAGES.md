# How to Pre-load Images

## Option 1: Put images in the `celebs` folder
1. Create a folder called `celebs` in your project root (if it doesn't exist)
2. Add image files (JPG, PNG) with names like: `celebrity_name.jpg` or `your_name.jpg`
3. The app will automatically load them on startup

## Option 2: Use the upload endpoint
POST to `/api/upload_image` with:
- `file`: image file
- `name`: name for the person (optional, defaults to filename)

## Option 3: Use the web interface
- Click "Add My Face" button
- Upload an image
- Enter a name

## Image Requirements:
- Must contain a clear face
- JPG or PNG format
- Will be automatically resized if too large
