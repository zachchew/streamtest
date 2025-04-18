import cv2
import time
from ultralytics import YOLO

# Replace with your IP Webcam's WireGuard VPN IP address and RTSP port/path
WIREGUARD_PEER_IP = "10.120.83.134"  # Example WG IP of your phone
RTSP_STREAM = f"http://{WIREGUARD_PEER_IP}:8080/video"  # IP Webcam RTSP format

# Load YOLO model
model = YOLO("model7.pt")  # Use a lightweight model like yolov8n for real-time inference

# Try connecting to RTSP stream
cap = cv2.VideoCapture(RTSP_STREAM)
if not cap.isOpened():
    print(f"❌ Failed to open RTSP stream at {RTSP_STREAM}")
    exit(1)
print(f"✅ Connected to RTSP stream at {RTSP_STREAM}")

# Inference frequency
target_hz = 5
frame_interval = 1.0 / target_hz

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame. Reconnecting might be needed.")
            continue

        # Inference
        results = model(frame)

        # Optional: save/log detections
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            coords = box.xyxy.numpy()[0]
            print(f"Class: {cls}, Conf: {conf:.2f}, Box: {coords}")

        # Optional: visualize (use only for debugging, not in headless server)
        # cv2.imshow("YOLO Detection", results[0].plot())
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Maintain 5 Hz rate
        elapsed = time.time() - start_time
        delay = frame_interval - elapsed
        if delay > 0:
            time.sleep(delay)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
