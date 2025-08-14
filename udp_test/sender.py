import socket
import time
import os

def send_image_udp(image_path, server_ip, server_port, chunk_size=1024):
    """
    Send a JPEG image via UDP in chunks
    """
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        print(f"Sending image: {image_path}")
        print(f"Image size: {len(image_data)} bytes")
        print(f"Server: {server_ip}:{server_port}")
        
        # Calculate number of chunks
        total_chunks = (len(image_data) + chunk_size - 1) // chunk_size
        
        # Send image in chunks
        for i in range(0, len(image_data), chunk_size):
            chunk = image_data[i:i + chunk_size]
            chunk_num = i // chunk_size
            
            # Create packet: [chunk_number:4 bytes][total_chunks:4 bytes][data]
            packet = chunk_num.to_bytes(4, 'big') + total_chunks.to_bytes(4, 'big') + chunk
            
            # Send chunk
            sock.sendto(packet, (server_ip, server_port))
            print(f"Sent chunk {chunk_num + 1}/{total_chunks} ({len(chunk)} bytes)")
            
            # Small delay to prevent overwhelming the receiver
            time.sleep(0.001)
        
        # Send end marker
        end_packet = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
        sock.sendto(end_packet, (server_ip, server_port))
        print("Image sent successfully!")
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "test_image.jpg"  # Change to your image path
    SERVER_IP = "192.168.1.100"   # Change to your laptop's IP
    SERVER_PORT = 9999
    CHUNK_SIZE = 1024
    
    send_image_udp(IMAGE_PATH, SERVER_IP, SERVER_PORT, CHUNK_SIZE)