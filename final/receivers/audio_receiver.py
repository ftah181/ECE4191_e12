import threading
import socket
import queue
import json
import time

class ADCReceiver(threading.Thread):
    def __init__(self, udp_port=5006):
        super().__init__(daemon=True)
        self.udp_port = udp_port
        self.running = True
        self.latest_voltage = 0.0
        self.data_queue = queue.Queue(maxsize=1024*5)
        
        # Setup UDP socket for ADC data
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(1)
            print(f"ADC UDP socket bound to port {udp_port}")
        except Exception as e:
            print(f"ADC socket binding error: {e}")
    
    def get_latest_voltage(self):
        return self.latest_voltage
    
    def get_voltage_data(self):
        """Get all queued voltage data"""
        data = []
        try:
            while True:
                data.append(self.data_queue.get_nowait())
        except queue.Empty:
            pass
        return data
    
    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                json_str = data.decode('utf-8')
                adc_data = json.loads(json_str)

                voltages = adc_data.get("voltages", [])
                timestamp = adc_data.get("timestamp", time.time())

                if voltages:
                    self.latest_voltage = voltages[-1]

                    for v in voltages:
                        entry = {"voltage": v, "timestamp": timestamp}
                        try:
                            self.data_queue.put_nowait(entry)
                        except queue.Full:
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(entry)
                            except:
                                pass

            except socket.timeout:
                continue
            except Exception as e:
                print(f"ADC receiver error: {e}")
                continue 
    
    def stop(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()