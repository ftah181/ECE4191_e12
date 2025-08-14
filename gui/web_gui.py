#!/usr/bin/env python3
"""
Web-based GUI - Single Channel with Animal Detection
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import json
import time
import threading
from collections import deque
import numpy as np
import random
from datetime import datetime
from inference_sdk import InferenceHTTPClient
import os

# Try to import hardware libraries. For testing when no hardware
try:
    import RPi.GPIO as GPIO
    import spidev
    REAL_HARDWARE = True
except ImportError:
    REAL_HARDWARE = False

app = Flask(__name__)

# Define web app class
class CameraSensorWebApp:
    def __init__(self):
        self.camera = None
        self.camera_running = False
        
        # Data storage - single channel only
        self.max_data_points = 100
        self.sensor_data = {
            'timestamps': deque(maxlen=self.max_data_points),
            'channel_0': deque(maxlen=self.max_data_points)
        }
        
        # Animal detection setup
        self.setup_animal_detection()
        self.detection_enabled = False
        self.current_frame = None
        self.predictions = []
        self.last_inference_time = 0
        self.inference_interval = 0.5  # Run inference every 0.5 seconds
        self.frame_lock = threading.Lock()
        
        self.setup_hardware()
        self.start_data_collection()
    
    def setup_animal_detection(self):
        """Initialize Roboflow client for animal detection"""
        try:
            self.detection_client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="lnHqcMh4NynT1If5FC38"  # Replace with your actual key
            )
            print("Animal detection client initialized")
        except Exception as e:
            print(f"Failed to initialize detection client: {e}")
            self.detection_client = None
    
    def setup_hardware(self):
        # Initialize SPI for ADC
        self.spi = None
        if REAL_HARDWARE:
            try:
                self.spi = spidev.SpiDev()
                self.spi.open(0, 0)
                self.spi.max_speed_hz = 1000000
                print("SPI initialized")
            except Exception as e:
                print(f"SPI init failed: {e}")
    
    def read_adc_channel(self, channel):
        # Read ADC channel
        if self.spi and channel < 8:
            try:
                adc_command = [1, (8 + channel) << 4, 0]
                adc_response = self.spi.xfer2(adc_command)
                adc_value = ((adc_response[1] & 3) << 8) + adc_response[2]
                return (adc_value / 1023.0) * 3.3
            except:
                return 0.0
        else:
            # Simulated data
            base_value = 1.5
            noise = random.uniform(-0.1, 0.1)
            return max(0, min(3.3, base_value + noise + 0.5 * np.sin(time.time())))
    
    def start_data_collection(self):
        # Start sensor data collection
        def collect_data():
            start_time = time.time()
            while True:
                try:
                    current_time = time.time() - start_time
                    self.sensor_data['timestamps'].append(current_time)
                    
                    # Read channel 0
                    value = self.read_adc_channel(0)
                    self.sensor_data['channel_0'].append(value)
                    
                    time.sleep(0.1)  # 10Hz sampling
                except Exception as e:
                    print(f"Data collection error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=collect_data)
        thread.daemon = True
        thread.start()
    
    def run_animal_detection(self, frame):
        """Run animal detection on a frame"""
        if not self.detection_client or not self.detection_enabled:
            return []
        
        try:
            # Save temporary image for inference
            temp_path = "IMG_3979.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Run inference
            result = self.detection_client.infer(temp_path, model_id="animal-detection-evlon/1")
            print(result)
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result.get("predictions", [])
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, predictions):
        """Draw bounding boxes and labels on frame"""
        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"]
            confidence = pred["confidence"]
            
            # Convert x, y, w, h to top-left and bottom-right points
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with background
            label = f"{class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 2)
    
    def start_camera(self):
        # Start camera capture
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_running = True
                
                # Start detection thread
                self.detection_thread = threading.Thread(target=self.detection_worker)
                self.detection_thread.daemon = True
                self.detection_thread.start()
                
                return True
        except Exception as e:
            print(f"Camera error: {e}")
        return False
    
    def stop_camera(self):
        # Stop camera
        self.camera_running = False
        if self.camera:
            self.camera.release()
    
    def detection_worker(self):
        """Background thread for running animal detection"""
        while self.camera_running:
            current_time = time.time()
            
            # Only run inference at specified intervals
            if (self.detection_enabled and 
                current_time - self.last_inference_time > self.inference_interval):
                
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_copy = self.current_frame.copy()
                
                predictions = self.run_animal_detection(frame_copy)
                
                with self.frame_lock:
                    self.predictions = predictions
                
                self.last_inference_time = current_time
            
            time.sleep(0.1)
    
    def generate_frames(self):
        # Generate camera frames
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Store current frame for detection
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        predictions_copy = self.predictions.copy()
                    
                    # Draw detections if enabled
                    if self.detection_enabled and predictions_copy:
                        self.draw_detections(frame, predictions_copy)
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    break
            except Exception as e:
                print(f"Frame generation error: {e}")
                break
    
    def get_sensor_data(self):
        # Get current sensor data
        return {
            'timestamps': list(self.sensor_data['timestamps'])[-50:],  # Last 50 points
            'channel_0': list(self.sensor_data['channel_0'])[-50:],
            'current_values': {
                'channel_0': list(self.sensor_data['channel_0'])[-1] if self.sensor_data['channel_0'] else 0,
            }
        }
    
    def get_detection_data(self):
        """Get current detection data"""
        with self.frame_lock:
            return {
                'enabled': self.detection_enabled,
                'predictions': self.predictions.copy(),
                'count': len(self.predictions)
            }
    
    def toggle_detection(self, enabled):
        """Enable/disable animal detection"""
        self.detection_enabled = enabled
        if not enabled:
            with self.frame_lock:
                self.predictions = []

# Global app instance
monitor_app = CameraSensorWebApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """Start camera endpoint"""
    success = monitor_app.start_camera()
    return jsonify({'success': success})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera endpoint"""
    monitor_app.stop_camera()
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(monitor_app.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensor_data')
def sensor_data():
    """Get sensor data"""
    return jsonify(monitor_app.get_sensor_data())

@app.route('/detection_data')
def detection_data():
    """Get detection data"""
    return jsonify(monitor_app.get_detection_data())

@app.route('/toggle_detection/<enabled>')
def toggle_detection(enabled):
    """Toggle animal detection on/off"""
    is_enabled = enabled.lower() == 'true'
    monitor_app.toggle_detection(is_enabled)
    return jsonify({'success': True, 'enabled': is_enabled})

@app.route('/save_data')
def save_data():
    """Save sensor data to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_data_{timestamp}.json"
        
        # Include both sensor and detection data
        sensor_data = monitor_app.get_sensor_data()
        detection_data = monitor_app.get_detection_data()
        
        combined_data = {
            'sensor_data': sensor_data,
            'detection_data': detection_data,
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# HTML template (create templates/index.html)
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Team E12 Web GUI with Animal Detection</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { display: flex; gap: 10px; margin: 10px 0; }
        .button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .button:hover { background: #0056b3; }
        .button.stop { background: #dc3545; }
        .button.stop:hover { background: #c82333; }
        .button.detection { background: #28a745; }
        .button.detection:hover { background: #218838; }
        .button.detection.disabled { background: #6c757d; }
        .values { display: flex; gap: 20px; margin: 20px 0; justify-content: center; flex-wrap: wrap; }
        .value-box { background: #e9ecef; padding: 25px; border-radius: 8px; text-align: center; min-width: 150px; }
        .value-number { font-size: 32px; font-weight: bold; color: #007bff; margin-top: 10px; }
        .detection-number { color: #28a745; }
        #camera-feed { max-width: 100%; height: 400px; background: #000; border-radius: 4px; }
        #sensor-plot { height: 400px; }
        .detection-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .detection-list { max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 14px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background-color: #28a745; }
        .status-inactive { background-color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Team E12 Web GUI with Animal Detection</h1>
        
        <!-- Camera Panel -->
        <div class="panel">
            <h2>Camera Feed</h2>
            <div class="controls">
                <button id="start-camera" class="button" onclick="startCamera()">Start Camera</button>
                <button id="stop-camera" class="button stop" onclick="stopCamera()">Stop Camera</button>
                <button id="toggle-detection" class="button detection" onclick="toggleDetection()">
                    <span class="status-indicator status-inactive" id="detection-status"></span>
                    Enable Detection
                </button>
            </div>
            <img id="camera-feed" src="/static/placeholder.jpg" alt="Camera feed will appear here">
            
            <!-- Detection Info -->
            <div class="detection-info">
                <h3>Animal Detection Status</h3>
                <div id="detection-results" class="detection-list">
                    Detection disabled. Click "Enable Detection" to start.
                </div>
            </div>
        </div>
        
        <!-- Sensor Panel -->
        <div class="panel">
            <h2>Audio Data & Detection Summary</h2>
            <div class="values">
                <div class="value-box">
                    <div style="font-size: 18px; color: #666;">Current Reading</div>
                    <div id="ch0-value" class="value-number">0.00 V</div>
                </div>
                <div class="value-box">
                    <div style="font-size: 18px; color: #666;">Animals Detected</div>
                    <div id="detection-count" class="value-number detection-number">0</div>
                </div>
            </div>
            <div class="controls">
                <button class="button" onclick="saveData()">Save Data</button>
                <button class="button" onclick="clearPlot()">Clear Plot</button>
            </div>
            <div id="sensor-plot"></div>
        </div>
    </div>

    <script>
        let detectionEnabled = false;
        
        // Initialize plot
        function initPlot() {
            const trace0 = { x: [], y: [], name: 'Channel 0', line: {color: 'red', width: 3} };
            
            const layout = {
                title: 'Microphone Channel',
                xaxis: { title: 'Time (s)' },
                yaxis: { title: 'Voltage (V)', range: [0, 3.3] }
            };
            
            Plotly.newPlot('sensor-plot', [trace0], layout);
        }

        // Update sensor data
        function updateSensorData() {
            fetch('/sensor_data')
                .then(response => response.json())
                .then(data => {
                    // Update current value
                    document.getElementById('ch0-value').textContent = data.current_values.channel_0.toFixed(2) + ' V';
                    
                    // Update plot
                    const update = {
                        x: [data.timestamps],
                        y: [data.channel_0]
                    };
                    Plotly.restyle('sensor-plot', update);
                });
        }
        
        // Update detection data
        function updateDetectionData() {
            fetch('/detection_data')
                .then(response => response.json())
                .then(data => {
                    // Update detection count
                    document.getElementById('detection-count').textContent = data.count;
                    
                    // Update detection results
                    const resultsDiv = document.getElementById('detection-results');
                    if (data.enabled && data.predictions.length > 0) {
                        let html = `Found ${data.predictions.length} animal(s):\\n\\n`;
                        data.predictions.forEach((pred, index) => {
                            html += `${index + 1}. ${pred.class} (${(pred.confidence * 100).toFixed(1)}%)\\n`;
                            html += `   Position: x=${pred.x.toFixed(0)}, y=${pred.y.toFixed(0)}\\n`;
                            html += `   Size: ${pred.width.toFixed(0)}x${pred.height.toFixed(0)}\\n\\n`;
                        });
                        resultsDiv.textContent = html;
                    } else if (data.enabled) {
                        resultsDiv.textContent = 'Detection active - no animals detected in current frame.';
                    } else {
                        resultsDiv.textContent = 'Detection disabled. Click "Enable Detection" to start.';
                    }
                });
        }

        // Camera controls
        function startCamera() {
            fetch('/start_camera')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('camera-feed').src = '/video_feed';
                    } else {
                        alert('Failed to start camera');
                    }
                });
        }

        function stopCamera() {
            fetch('/stop_camera');
            document.getElementById('camera-feed').src = '/static/placeholder.jpg';
            // Also disable detection when camera stops
            if (detectionEnabled) {
                toggleDetection();
            }
        }
        
        function toggleDetection() {
            const newState = !detectionEnabled;
            fetch(`/toggle_detection/${newState}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        detectionEnabled = data.enabled;
                        updateDetectionButton();
                    }
                });
        }
        
        function updateDetectionButton() {
            const button = document.getElementById('toggle-detection');
            const status = document.getElementById('detection-status');
            
            if (detectionEnabled) {
                button.innerHTML = '<span class="status-indicator status-active" id="detection-status"></span>Disable Detection';
                status.className = 'status-indicator status-active';
            } else {
                button.innerHTML = '<span class="status-indicator status-inactive" id="detection-status"></span>Enable Detection';
                status.className = 'status-indicator status-inactive';
            }
        }

        function saveData() {
            fetch('/save_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Data saved to ' + data.filename);
                    } else {
                        alert('Failed to save data: ' + data.error);
                    }
                });
        }

        function clearPlot() {
            const update = {
                x: [[]],
                y: [[]]
            };
            Plotly.restyle('sensor-plot', update);
        }

        // Initialize and start updates
        initPlot();
        setInterval(updateSensorData, 1000); // Update every second
        setInterval(updateDetectionData, 1000); // Update detection every second
        updateDetectionButton(); // Initialize button state
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Create templates directory and file
    import os
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("Starting web server with animal detection...")
    print("Open http://your_pi_ip:5000 in your browser")
    print("Or http://localhost:5000 if running locally")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)