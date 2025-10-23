# Footfall Counter

A simple web app that counts people entering and exiting in videos using computer vision.

## What it does

Upload a video file and watch it play with live counting. When the video ends, you'll see how many people went in and out.

## How to run it

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python counter_stream.py
   ```

3. Open your browser and go to `http://127.0.0.1:5000/`

## How to use it

1. Click "Choose File" and pick a video
2. Click "Start Processing"
3. Watch the video with live counting
4. See the final results when it's done

## How it works

The app uses background subtraction to detect moving objects and tracks their direction to count people going in and out. It works best with videos that have clear entry/exit points and good lighting.

## Files

- `counter_stream.py` - Main application
- `templates/index.html` - Web interface
- `requirements.txt` - Required Python packages

## Notes

This is a basic implementation for learning purposes. The counting accuracy depends on video quality and lighting conditions.
