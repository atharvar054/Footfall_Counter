import time
import cv2 
from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import imutils
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import io
import base64


app = Flask(__name__)

# Global variables for video processing
current_video_path = 'video.mp4'  # Default video
video_processing_complete = False
final_counts = {'in_count': 0, 'out_count': 0, 'total_count': 0}

# Global variables for visualization
trajectories = defaultdict(lambda: deque(maxlen=50))  # Store last 50 points per trajectory
heatmap_data = np.zeros((480, 500), dtype=np.float32)  # Heatmap accumulator
show_heatmap = False
show_trajectories = False

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload."""
    global current_video_path, video_processing_complete, final_counts
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video file selected'}), 400
        
        # Save uploaded file
        video_file.save('uploaded_video.mp4')
        current_video_path = 'uploaded_video.mp4'
        
        # Reset processing state
        video_processing_complete = False
        final_counts = {'in_count': 0, 'out_count': 0, 'total_count': 0}
        
        return jsonify({'success': True, 'message': 'Video uploaded successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_status')
def video_status():
    """Check if video processing is complete."""
    global video_processing_complete, final_counts
    
    return jsonify({
        'completed': video_processing_complete,
        'counts': final_counts
    })

@app.route('/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    """Toggle heatmap visualization."""
    global show_heatmap
    show_heatmap = not show_heatmap
    return jsonify({'show_heatmap': show_heatmap})

@app.route('/toggle_trajectories', methods=['POST'])
def toggle_trajectories():
    """Toggle trajectory visualization."""
    global show_trajectories
    show_trajectories = not show_trajectories
    return jsonify({'show_trajectories': show_trajectories})

@app.route('/reset_visualization', methods=['POST'])
def reset_visualization():
    """Reset heatmap and trajectories."""
    global heatmap_data, trajectories
    heatmap_data = np.zeros((480, 500), dtype=np.float32)
    trajectories.clear()
    return jsonify({'success': True})

def create_heatmap_overlay(heatmap_data, alpha=0.3):
    """Create a heatmap overlay for the frame."""
    # Normalize heatmap data
    if np.max(heatmap_data) > 0:
        normalized_heatmap = heatmap_data / np.max(heatmap_data)
    else:
        normalized_heatmap = heatmap_data
    
    # Create colormap
    cmap = cm.get_cmap('hot')
    heatmap_colored = cmap(normalized_heatmap)
    
    # Convert to BGR for OpenCV
    heatmap_bgr = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
    
    return heatmap_bgr

def draw_trajectories(frame, trajectories):
    """Draw trajectory paths on the frame."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for i, (trajectory_id, points) in enumerate(trajectories.items()):
        if len(points) > 1:
            color = colors[i % len(colors)]
            # Draw trajectory line
            for j in range(1, len(points)):
                pt1 = (int(points[j-1][0]), int(points[j-1][1]))
                pt2 = (int(points[j][0]), int(points[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw current position
            if points:
                current_pos = (int(points[-1][0]), int(points[-1][1]))
                cv2.circle(frame, current_pos, 5, color, -1)
                cv2.putText(frame, f"T{trajectory_id}", 
                           (current_pos[0] + 10, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def find_max(k):
    d = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in d: 
            d[n] += 1
        else: 
            d[n] = 1

        # Keep track of maximum on the go
        if d[n] > maximum[1]: 
            maximum = (n,d[n])

    return maximum

def gen():
    """Video streaming generator function with heatmap and trajectory visualization."""
    global current_video_path, video_processing_complete, final_counts
    global trajectories, heatmap_data, show_heatmap, show_trajectories
    
    cap = cv2.VideoCapture(current_video_path)
    avg = None
    xvalues = list()
    motion = list()
    count1 = 0
    count2 = 0
    trajectory_id = 0

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video ended, update final counts
            video_processing_complete = True
            final_counts = {
                'in_count': count1,
                'out_count': count2,
                'total_count': count1 + count2
            }
            break
            
        flag = True
        frame = imutils.resize(frame, width=500)
        height, width = frame.shape[:2]
        
        # Resize heatmap if needed
        if heatmap_data.shape != (height, width):
            heatmap_data = np.zeros((height, width), dtype=np.float32)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
        if avg is None:
            avg = gray.copy().astype("float")
            continue
    
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 2, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centroids = []
        
        for c in cnts:
            if cv2.contourArea(c) < 5000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            xvalues.append(x)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Calculate centroid for trajectory tracking
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            current_centroids.append((centroid_x, centroid_y))
            
            # Update heatmap data
            if show_heatmap:
                # Add Gaussian kernel to heatmap
                kernel_size = 15
                sigma = 5
                y1 = max(0, centroid_y - kernel_size//2)
                y2 = min(height, centroid_y + kernel_size//2)
                x1 = max(0, centroid_x - kernel_size//2)
                x2 = min(width, centroid_x + kernel_size//2)
                
                # Create Gaussian kernel
                gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
                gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
                
                # Add to heatmap
                heatmap_data[y1:y2, x1:x2] += gaussian_kernel[:y2-y1, :x2-x1]
            
            flag = False
        
        # Track trajectories
        if show_trajectories and current_centroids:
            # Simple trajectory tracking - assign new ID to each detected object
            for centroid in current_centroids:
                trajectories[trajectory_id].append(centroid)
                trajectory_id += 1
    	
        no_x = len(xvalues)
        
        if (no_x > 2):
            difference = xvalues[no_x - 1] - xvalues[no_x - 2]
            if(difference > 0):
                motion.append(1)
            else:
                motion.append(0)
    
        if flag is True:
            if no_x > 5:
                val, times = find_max(motion)
                if val == 1 and times >= 15:
                    count1 += 1
                else:
                    count2 += 1
                    
            xvalues = list()
            motion = list()
        
        # Draw counting lines
        cv2.line(frame, (200, 0), (200, height), (0, 255, 0), 2)
        cv2.line(frame, (240, 0), (240, height), (0, 255, 0), 2)
        
        # Draw trajectories if enabled
        if show_trajectories:
            draw_trajectories(frame, trajectories)
        
        # Draw heatmap overlay if enabled
        if show_heatmap:
            heatmap_overlay = create_heatmap_overlay(heatmap_data)
            # Blend heatmap with frame
            frame = cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.3, 0)
        
        # Draw counters and status
        cv2.putText(frame, "In: {}".format(count1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Out: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw visualization status
        y_offset = 60
        if show_heatmap:
            cv2.putText(frame, "Heatmap: ON", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        if show_trajectories:
            cv2.putText(frame, "Trajectories: ON", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        
        cv2.imshow("Frame", frame)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

