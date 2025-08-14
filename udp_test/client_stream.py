#!/usr/bin/env python3
"""
UDP Video Client for Laptop
Receives video stream from Raspberry Pi via UDP
"""

import cv2
import socket
import pickle
import struct
import numpy as np
import threading
import time
from collections import defaultdict

class UDPVideoClient:
    def __init__(self, server_ip, server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(5.0)
        self.running = False
        self.frame_buffer = {}
        self.current_frame_chunks = defaultdict(dict)
        
    def connect_to_server(self):
        """Send connection request to server"""
        try:
            self.socket.sendto(b'CONNECT', (self.server_ip, self.server_port))
            print(f"Connected to server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def disconnect_from_server(self):
        """Send disconnection request to server"""
        try:
            self.socket.sendto(b'DISCONNECT', (self.server_ip, self.server_port))
        except:
            pass
    
    def receive_frame_chunks(self):
        """Receive and reassemble frame chunks"""
        while self.running:
            try:
                # Receive header with chunk info
                header_data, addr = self.socket.recvfrom(8)
                if len(header_data) == 8:
                    total_chunks, total_size = struct.unpack('!II', header_data)
                    
                    chunks = {}
                    # Receive all chunks for this frame
                    for _ in range(total_chunks):
                        chunk_data, _ = self.socket.recvfrom(65507)
                        chunk_id = struct.unpack('!I', chunk_data[:4])[0]
                        chunks[chunk_id] = chunk_data[4:]
                    
                    # Reassemble frame
                    if len(chunks) == total_chunks:
                        frame_data = b''.join([chunks[i] for i in range(total_chunks)])
                        if len(frame_data) == total_size:
                            self.frame_buffer['latest'] = frame_data
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error receiving frame: {e}")
                break
    
    def display_video(self):
        """Display received video frames"""
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            if 'latest' in self.frame_buffer:
                try:
                    # Decode frame
                    frame_data = self.frame_buffer['latest']
                    buffer = pickle.loads(frame_data)
                    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                    
                    # TODO: Add  classification here
                    #  Run classification on "frame"

                    if frame is not None:
                        # Add frame info
                        frame_count += 1
                        current_time = time.time()
                        if current_time - last_time >= 1.0:
                            fps = frame_count / (current_time - last_time)
                            print(f"FPS: {fps:.1f}")
                            frame_count = 0
                            last_time = current_time
                        
                        # Display frame
                        # TODO: change to display classification
                        cv2.imshow('UDP Video Stream', frame)
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop()
                            break
                    
                    # Clear the buffer to get fresh frames
                    del self.frame_buffer['latest']
                    
                except Exception as e:
                    print(f"Error decoding frame: {e}")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    
    def start(self):
        """Start the video client"""
        if not self.connect_to_server():
            return
        
        self.running = True
        
        # Start frame receiver thread
        receiver_thread = threading.Thread(target=self.receive_frame_chunks)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        print("Starting video display... Press 'q' to quit")
        
        try:
            self.display_video()
        except KeyboardInterrupt:
            print("\nShutting down client...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the client and cleanup"""
        self.running = False
        self.disconnect_from_server()
        cv2.destroyAllWindows()
        self.socket.close()
        print("Client stopped")

if __name__ == "__main__":
    # Replace with your Raspberry Pi's IP address
    PI_IP = input("Enter Raspberry Pi IP address: ").strip()
    if not PI_IP:
        PI_IP = "192.168.1.100"  # Default example
    
    client = UDPVideoClient(PI_IP)
    client.start()