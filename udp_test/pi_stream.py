#!/usr/bin/env python3
"""
UDP Video Server for Raspberry Pi
Captures video from camera and streams it via UDP
"""

import cv2
import socket
import pickle
import struct
import threading
import time

class UDPVideoServer:
    def __init__(self, host='0.0.0.0', port=9999, max_packet_size=65507):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.clients = set()
        self.running = False
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        
    def handle_client_requests(self):
        """Handle incoming client connections"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                if data == b'CONNECT':
                    self.clients.add(addr)
                    print(f"Client connected: {addr}")
                elif data == b'DISCONNECT':
                    self.clients.discard(addr)
                    print(f"Client disconnected: {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error handling client request: {e}")
    
    def send_frame_chunks(self, frame_data, client_addr):
        """Send frame data in chunks to avoid UDP packet size limits"""
        total_size = len(frame_data)
        chunk_size = self.max_packet_size - 100  # Leave room for headers
        
        # Send total chunks info first
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        header = struct.pack('!II', total_chunks, total_size)
        self.socket.sendto(header, client_addr)
        
        # Send chunks
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            chunk = frame_data[start:end]
            chunk_header = struct.pack('!I', i)
            self.socket.sendto(chunk_header + chunk, client_addr)
    
    def stream_video(self):
        """Capture and stream video frames"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Compress frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_data = pickle.dumps(buffer)
            
            # Send to all connected clients
            disconnected_clients = []
            for client_addr in self.clients.copy():
                try:
                    self.send_frame_chunks(frame_data, client_addr)
                except Exception as e:
                    print(f"Error sending to {client_addr}: {e}")
                    disconnected_clients.append(client_addr)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.clients.discard(client)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Streamed {frame_count} frames to {len(self.clients)} clients")
            
            time.sleep(0.05)  # ~20 FPS
    
    def start(self):
        """Start the video streaming server"""
        self.running = True
        self.socket.settimeout(1.0)
        
        print(f"UDP Video Server starting on {self.host}:{self.port}")
        
        # Start client handler thread
        client_thread = threading.Thread(target=self.handle_client_requests)
        client_thread.daemon = True
        client_thread.start()
        
        try:
            self.stream_video()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server and cleanup resources"""
        self.running = False
        self.cap.release()
        self.socket.close()
        print("Server stopped")

if __name__ == "__main__":
    server = UDPVideoServer()
    server.start()