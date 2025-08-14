#!/usr/bin/env python3
"""
Web-based GUI - Single Channel
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
        
        self.setup_hardware()
        self.start_data_collection()
    
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
    
    def start_camera(self):
        # Start camera capture
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_running = True
                return True
        except Exception as e:
            print(f"Camera error: {e}")
        return False
    
    def stop_camera(self):
        # Stop camera
        self.camera_running = False
        if self.camera:
            self.camera.release()
    
    def generate_frames(self):
        # Generate camera frames
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
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

@app.route('/save_data')
def save_data():
    """Save sensor data to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_data_{timestamp}.json"
        
        data = monitor_app.get_sensor_data()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# HTML template (create templates/index.html)
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pi Camera & Single Sensor Monitor</title>
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
        .values { display: flex; gap: 20px; margin: 20px 0; justify-content: center; }
        .value-box { background: #e9ecef; padding: 25px; border-radius: 8px; text-align: center; min-width: 150px; }
        .value-number { font-size: 32px; font-weight: bold; color: #007bff; margin-top: 10px; }
        #camera-feed { max-width: 100%; height: 400px; background: #000; border-radius: 4px; }
        #sensor-plot { height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pi Camera & Single Sensor Monitor</h1>
        
        <!-- Camera Panel -->
        <div class="panel">
            <h2>Camera Feed</h2>
            <div class="controls">
                <button id="start-camera" class="button" onclick="startCamera()">Start Camera</button>
                <button id="stop-camera" class="button stop" onclick="stopCamera()">Stop Camera</button>
            </div>
            <img id="camera-feed" src="/static/placeholder.jpg" alt="Camera feed will appear here">
        </div>
        
        <!-- Sensor Panel -->
        <div class="panel">
            <h2>Sensor Data (Channel 0)</h2>
            <div class="values">
                <div class="value-box">
                    <div style="font-size: 18px; color: #666;">Current Reading</div>
                    <div id="ch0-value" class="value-number">0.00 V</div>
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
        // Initialize plot
        function initPlot() {
            const trace0 = { x: [], y: [], name: 'Channel 0', line: {color: 'red', width: 3} };
            
            const layout = {
                title: 'Real-time Sensor Data (Channel 0)',
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
    
    print("Starting web server...")
    print("Open http://your_pi_ip:5000 in your browser")
    print("Or http://localhost:5000 if running locally")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)