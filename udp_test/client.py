import socket
import time
import os

def receive_image_udp(port, output_path="received_image.jpg", timeout=30):
    """
    Receive a JPEG image via UDP and save it
    """
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    
    try:
        # Bind to port
        sock.bind(('', port))
        print(f"Server listening on port {port}")
        print("Waiting for image...")
        
        chunks = {}
        total_chunks = None
        start_time = time.time()
        
        while True:
            try:
                # Receive packet
                data, addr = sock.recvfrom(65536)  # Max UDP packet size
                
                # Check for end marker
                if data == b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF':
                    print("End marker received")
                    break
                
                # Parse packet
                if len(data) < 8:
                    continue
                    
                chunk_num = int.from_bytes(data[:4], 'big')
                total_chunks_received = int.from_bytes(data[4:8], 'big')
                chunk_data = data[8:]
                
                # Store chunk
                chunks[chunk_num] = chunk_data
                
                if total_chunks is None:
                    total_chunks = total_chunks_received
                    print(f"Expecting {total_chunks} chunks")
                
                print(f"Received chunk {chunk_num + 1}/{total_chunks} ({len(chunk_data)} bytes) from {addr[0]}")
                
                # Check if we have all chunks
                if len(chunks) >= total_chunks:
                    print("All chunks received!")
                    break
                    
            except socket.timeout:
                print("Timeout waiting for data")
                break
        
        # Reconstruct image
        if chunks and total_chunks and len(chunks) >= total_chunks:
            print("Reconstructing image...")
            image_data = b''
            
            # Combine chunks in order
            for i in range(total_chunks):
                if i in chunks:
                    image_data += chunks[i]
                else:
                    print(f"Warning: Missing chunk {i}")
            
            # Save image
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            elapsed_time = time.time() - start_time
            print(f"Image saved as '{output_path}'")
            print(f"Total size: {len(image_data)} bytes")
            print(f"Transfer time: {elapsed_time:.2f} seconds")
            
        else:
            print("Failed to receive complete image")
            print(f"Received {len(chunks)}/{total_chunks or 'unknown'} chunks")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    # Configuration
    PORT = 9999
    OUTPUT_PATH = "received_image.jpg"
    TIMEOUT = 30  # seconds
    
    receive_image_udp(PORT, OUTPUT_PATH, TIMEOUT)