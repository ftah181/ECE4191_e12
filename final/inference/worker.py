import threading
import queue
from inference.model import YOLOModel

class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=3)
        self.running = True
        self.model = YOLOModel()
        
    def add_frame(self, frame):
        try:
            # Always clear queue and add latest frame only
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                
                # Run inference
                processed_frame, predictions = self.model.predict(frame, force_inference=True)
                
                # Keep only latest result
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                try:
                    self.result_queue.put_nowait((processed_frame, predictions))
                except queue.Full:
                    pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference worker error: {e}")
                continue
    
    def stop(self):
        self.running = False