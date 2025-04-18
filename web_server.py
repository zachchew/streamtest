import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template
from flask_cors import CORS

# Replace with your IP Webcam's WireGuard VPN IP address and RTSP port/path
WIREGUARD_PEER_IP = "10.120.83.146"  # Example WG IP of your phone
RTSP_STREAM = f"http://{WIREGUARD_PEER_IP}:8080/video"  # IP Webcam RTSP format

# Load YOLO model
model = YOLO("model1.pt")  # Use your trained model

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# Global variables
frame_lock = threading.Lock()
current_frame = None
processing = True

# Flask app for web streaming
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def process_stream():
    """Background thread function to process the RTSP stream with YOLO."""
    global current_frame, processing
    
    # Connect to RTSP stream
    cap = cv2.VideoCapture(RTSP_STREAM)
    if not cap.isOpened():
        print(f"❌ Failed to open RTSP stream at {RTSP_STREAM}")
        return
    
    print(f"✅ Connected to RTSP stream at {RTSP_STREAM}")
    print(f"✅ Using confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Inference frequency
    target_hz = 5
    frame_interval = 1.0 / target_hz
    
    try:
        while processing:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame. Reconnecting might be needed.")
                time.sleep(1)
                continue
            
            # Run YOLO inference
            results = model(frame)
            
            # Instead of using the built-in plot method, we'll draw our own boxes
            # based on the confidence threshold
            annotated_frame = frame.copy()
            
            # Get detection results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get confidence
                    conf = float(box.conf)
                    
                    # Only show boxes with confidence > threshold
                    if conf > CONFIDENCE_THRESHOLD:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get class and class name
                        cls = int(box.cls)
                        cls_name = result.names[cls]
                        
                        # Draw rectangle and label
                        color = (0, 255, 0)  # Green color for boxes
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with confidence
                        label = f"{cls_name}: {conf:.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        # Log the detection
                        print(f"Detection: Class: {cls_name}, Conf: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
            
            # Update the current frame with the lock to avoid race conditions
            with frame_lock:
                # Convert to JPEG for streaming
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                current_frame = buffer.tobytes()
            
            # Maintain target rate
            elapsed = time.time() - start_time
            delay = frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)
                
    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        cap.release()
        print("Video processing stopped")

def generate_frames():
    """Generator function for MJPEG streaming."""
    global current_frame
    while True:
        # Get the current frame with lock
        with frame_lock:
            if current_frame is None:
                # Create blank frame if no frame is available
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for video...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, blank_buffer = cv2.imencode('.jpg', blank_frame)
                blank_bytes = blank_buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
                time.sleep(0.1)
                continue
                
            # Send the current frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')

@app.route('/')
def index():
    """Serve a simple HTML page for debugging."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the video stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/settings')
def get_settings():
    """API endpoint to get current settings."""
    return {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'stream_fps': 5,
        'model': 'best.pt'
    }

if __name__ == '__main__':
    # Start the video processing in a background thread
    video_thread = threading.Thread(target=process_stream)
    video_thread.daemon = True
    video_thread.start()
    
    try:
        # Start the Flask server
        print("🚀 Starting Flask server on port 5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # Cleanup when server is shut down
        processing = False
        video_thread.join(timeout=2)
        print("Server shutting down")