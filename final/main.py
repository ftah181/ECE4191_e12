import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import datetime

# Import Threads
from receivers.video_receiver import VideoReceiver
from receivers.audio_receiver import ADCReceiver
from inference.worker import InferenceWorker
from utils.config import *
from utils.helpers import resize_for_display

class GUI:
    def __init__(self, root, udp_ip=DEFAULT_UDP_IP, video_port=DEFAULT_VIDEO_PORT, adc_port=DEFAULT_ADC_PORT):
        self.root = root
        self.root.title("UDP Video + ADC Display")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2C3E50')

        # Performance settings
        self.skip_inference = tk.BooleanVar(value=False)
        
        # Setup receiver threads
        self.video_receiver = VideoReceiver(udp_ip, video_port)
        self.video_receiver.start()
        
        self.adc_receiver = ADCReceiver(adc_port)
        self.adc_receiver.start()

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Threading for inference
        self.inference_worker = InferenceWorker()
        self.inference_worker.start()
        self.last_inference_result = None
        
        # Setup UI
        self._setup_ui()
        
        # Setup recording
        self.video_out = None
        self.recording = False
        self.video_filename = "output.mp4"
        
        # Setup data tracking
        self.frame_count = 0
        self.voltage_update_counter = 0
        self.plot_time_offset = time.time()
        
        # Detection tracking
        self.detection_list = []
        self.detection_counter = 0
        self.detected_animals = set()
        
        # Start main update loop
        self.update()

    def _setup_ui(self):
        """Setup the user interface components"""
        # Main layout
        self.left_frame = tk.Frame(self.root, bg='#2C3E50')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        self.right_frame = tk.Frame(self.root, bg='#34495E', width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_frame.pack_propagate(False)
        
        self._setup_detection_panel()
        self._setup_video_panel()
        self._setup_graph_panel()
        self._setup_spectrogram_panel()
        self._setup_controls()

    def _setup_detection_panel(self):
        """Setup animal detection list panel"""
        # Animal detection list
        self.detection_title = tk.Label(self.right_frame, text="Detected Animals", 
                                       font=("Arial", 14, "bold"), bg='#34495E', fg='white')
        self.detection_title.pack(pady=(10, 5))
        
        # Scrollable frame for animal list
        self.detection_section = tk.Frame(self.right_frame, bg='#34495E')
        self.detection_section.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
        self.detection_canvas = tk.Canvas(self.detection_section, bg='#34495E', highlightthickness=0)
        self.detection_scrollbar = tk.Scrollbar(self.detection_section, orient="vertical", command=self.detection_canvas.yview)
        self.detection_scrollable_frame = tk.Frame(self.detection_canvas, bg='#34495E')

        self.detection_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.detection_canvas.configure(scrollregion=self.detection_canvas.bbox("all"))
        )
        
        self.detection_canvas.create_window((0, 0), window=self.detection_scrollable_frame, anchor="nw")
        self.detection_canvas.configure(yscrollcommand=self.detection_scrollbar.set)
        
        self.detection_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=5)
        self.detection_scrollbar.pack(side="right", fill="y", pady=5)
        
        # Clear button
        self.clear_button = tk.Button(self.detection_section, text="Clear List", 
                                     command=self.clear_detection_list,
                                     bg='#E74C3C', fg='white', font=("Arial", 10))
        self.clear_button.pack(pady=5)

    def _setup_video_panel(self):
        """Setup video display panel"""
        self.video_frame = tk.Frame(self.left_frame, bg='#2C3E50')
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Status labels
        self.status_label = tk.Label(self.video_frame, text="Waiting for UDP frames...", font=("Arial", 10))
        self.status_label.pack()
        
        self.fps_label = tk.Label(self.video_frame, text="FPS: 0", font=("Arial", 10), fg="blue")
        self.fps_label.pack()
        
        self.adc_status_label = tk.Label(self.video_frame, text="ADC: Waiting...", font=("Arial", 10), fg="green")
        self.adc_status_label.pack()

        # Video display
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Frame counter
        self.frame_counter_label = tk.Label(self.video_frame, text="Frames: 0")
        self.frame_counter_label.pack()

    def _setup_graph_panel(self):
        """Setup voltage graph panel"""
        self.graph_frame = tk.Frame(self.left_frame, bg='#2C3E50')
        self.graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Voltage Graph Setup
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('#2C3E50')
        self.ax.set_facecolor('#34495E')
        self.ax.set_ylim(-0.01, 0.01)
        self.ax.set_title("Microphone Data", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Magnitude", color='white')
        self.ax.tick_params(colors='white')
        self.x_data = []
        self.y_data = []
        self.time_window = 5  
        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_spectrogram_panel(self):
        """Setup spectrogram panel"""
        self.spec_frame = tk.Frame(self.right_frame, bg='#2C3E50', height=250)
        self.spec_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Spectrogram setup
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(6, 3))
        self.ax_spec.set_title("Spectrogram")
        self.ax_spec.set_xlabel("Time [s]")
        self.ax_spec.set_ylabel("Frequency [Hz]")
        self.spectogram_queue = []
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.spec_frame)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_controls(self):
        """Setup control buttons and checkboxes"""
        # Performance controls
        self.controls_frame = tk.Frame(self.video_frame)
        self.controls_frame.pack()
        
        tk.Checkbutton(self.controls_frame, text="Skip Inference", 
                      variable=self.skip_inference).pack(side=tk.LEFT)

        # Recording button
        self.record_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)

    def toggle_recording(self):
        """Toggle video recording on/off"""
        if not self.recording:
            self.recording = True
            self.record_button.config(text="Stop Recording")
            print("Recording started")
        else:
            self.recording = False
            self.record_button.config(text="Start Recording")
            if self.video_out is not None:
                self.video_out.release()
                self.video_out = None
            print("Recording stopped")
    
    def calculate_fps(self):
        """Calculate and update FPS display"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.display_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.fps_label.config(text=f"FPS: {self.display_fps:.1f}")
    
    def update_detection_list(self, predictions):
        """Update the animal detection list with new predictions"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        for pred in predictions:
            if pred["confidence"] > 0.5:
                animal_name = pred["class"].lower()
                
                if animal_name not in self.detected_animals:
                    self.detected_animals.add(animal_name)
                    self.detection_counter += 1
                    confidence = pred["confidence"]
                    
                    # Create detection entry
                    detection_frame = tk.Frame(self.detection_scrollable_frame, bg='#2C3E50', 
                                             relief=tk.RAISED, bd=1)
                    detection_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    # Detection info
                    info_text = f"#{self.detection_counter}: {pred['class']}\nConfidence: {confidence:.2f}\nTime: {current_time}"
                    detection_label = tk.Label(detection_frame, text=info_text, 
                                             bg='#2C3E50', fg='white', font=("Arial", 9),
                                             justify=tk.LEFT)
                    detection_label.pack(padx=5, pady=3)
                    
                    # Store detection info
                    self.detection_list.append({
                        'frame': detection_frame,
                        'animal': pred['class'],
                        'confidence': confidence,
                        'time': current_time
                    })
                    
                    # Auto-scroll to bottom
                    self.detection_canvas.update_idletasks()
                    self.detection_canvas.yview_moveto(1.0)
    
    def clear_detection_list(self):
        """Clear all detections from the list"""
        for detection in self.detection_list:
            detection['frame'].destroy()
        self.detection_list.clear()
        self.detection_counter = 0
        self.detected_animals.clear()

    def update(self):
        """Main update loop"""
        # Process video frames
        self._process_video_frames()
        
        # Update voltage plot
        self._update_voltage_plot()
        
        # Schedule next update
        self.root.after(5, self.update)

    def _process_video_frames(self):
        """Process incoming video frames"""
        new_frame = self.video_receiver.get_latest_frame()
        
        if new_frame is not None:
            self.frame_count += 1
            self.status_label.config(text=f"Video UDP:")
            
            try:
                if self.skip_inference.get():
                    processed_frame = new_frame
                else:
                    # Pass frame to inference thread
                    self.inference_worker.add_frame(new_frame)
                    
                    # Check for completed inference
                    result = self.inference_worker.get_result()
                    if result is not None:
                        self.last_inference_result = result
                        # Update detection list
                        _, predictions = result
                        self.update_detection_list(predictions)
                    
                    # Use last result or original frame
                    if self.last_inference_result is not None:
                        processed_frame, _ = self.last_inference_result
                    else:
                        processed_frame = new_frame
                
                # Display frame
                display_frame = resize_for_display(processed_frame)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Handle recording
                self._handle_recording(display_frame)

                # Update counters
                self.frame_counter_label.config(text=f"Frames: {self.frame_count}")
                self.calculate_fps()
                
            except Exception as e:
                print(f"Frame processing error: {e}")

    def _handle_recording(self, frame):
        """Handle video recording"""
        if self.recording:
            if self.video_out is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_out = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (w, h))
                
                if not self.video_out.isOpened():
                    print("Failed to initialize VideoWriter")
                    self.video_out = None
        
            if self.video_out is not None:
                self.video_out.write(frame)

    def _update_voltage_plot(self):
        """Update voltage plot and spectrogram"""
        self.voltage_update_counter += 1
        if self.voltage_update_counter >= 10:
            self.voltage_update_counter = 0
            
            try:
                voltage_data_list = self.adc_receiver.get_voltage_data()
                
                if voltage_data_list:
                    for voltage_data in voltage_data_list:
                        voltage = voltage_data['voltage']
                        current_time = time.time() - self.plot_time_offset
                        
                        if len(self.x_data) >= PLOT_X_LENGTH:
                            self.x_data = self.x_data[1:]
                            self.y_data = self.y_data[1:]
                        
                        self.x_data.append(current_time)
                        self.y_data.append(voltage)
                    
                    # Update status
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    
                    # Update plot
                    if len(self.x_data) > 0:
                        while self.x_data and (self.x_data[-1] - self.x_data[0]) > self.time_window:
                            self.x_data.pop(0)
                            self.y_data.pop(0)

                        self.line.set_data(self.x_data, self.y_data)
                        if len(self.x_data) > 1:
                            self.ax.set_xlim(self.x_data[0], self.x_data[0] + self.time_window)

                        self.canvas.draw_idle()
                else:
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    if latest_voltage > 0:
                        self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    else:
                        self.adc_status_label.config(text="ADC: Waiting for UDP data...")
                        
            except Exception as e:
                print(f"Plot update error: {e}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'inference_worker'):
            self.inference_worker.stop()
        if hasattr(self, 'adc_receiver'):
            self.adc_receiver.stop()
        if hasattr(self, 'video_receiver'):
            self.video_receiver.stop()
        if self.video_out is not None:
            self.video_out.release()

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if hasattr(app, 'inference_worker'):
            app.inference_worker.stop()
        if hasattr(app, 'adc_receiver'):
            app.adc_receiver.stop()
        if hasattr(app, 'video_receiver'):
            app.video_receiver.stop()