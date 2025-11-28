import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
import threading
from datetime import datetime
import os

from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA
from uago_utils import (
    setup_logging, load_config, save_config,
    save_phase_data, save_full_cycle,
    process_image_input, process_video_input,
    capture_from_webcam, create_roi_overlay
)

class UAGOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UAGO - Universal Adaptive Geometric Observer")
        self.root.geometry("1200x800")

        self.config = load_config()
        self.api_key = self.config.get('api_key', '')
        self.demo_mode = not bool(self.api_key)

        self.current_frame = None
        self.cycle_data = None
        self.processing = False

        self.setup_ui()
        self.logger = setup_logging()

    def setup_ui(self):
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(main_container)
        right_panel = ttk.Frame(main_container)

        main_container.add(left_panel, weight=1)
        main_container.add(right_panel, weight=2)

        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)

    def setup_left_panel(self, parent):
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(config_frame, text="Mistral API Key:").pack(anchor=tk.W)
        self.api_key_entry = ttk.Entry(config_frame, show="*", width=40)
        self.api_key_entry.insert(0, self.api_key)
        self.api_key_entry.pack(fill=tk.X, pady=5)

        ttk.Button(config_frame, text="Save API Key", command=self.save_api_key).pack(pady=5)

        self.demo_var = tk.BooleanVar(value=self.demo_mode)
        ttk.Checkbutton(
            config_frame,
            text="Demo Mode (use preloaded data)",
            variable=self.demo_var
        ).pack(anchor=tk.W, pady=5)

        input_frame = ttk.LabelFrame(parent, text="Input Selection", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_type = tk.StringVar(value="image")

        ttk.Radiobutton(
            input_frame,
            text="Upload Image",
            variable=self.input_type,
            value="image"
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            input_frame,
            text="Upload Video",
            variable=self.input_type,
            value="video"
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            input_frame,
            text="Webcam Capture",
            variable=self.input_type,
            value="webcam"
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            input_frame,
            text="Use Demo",
            variable=self.input_type,
            value="demo"
        ).pack(anchor=tk.W, pady=2)

        ttk.Button(
            input_frame,
            text="📁 Select File / Capture",
            command=self.select_input
        ).pack(fill=tk.X, pady=10)

        self.preview_label = ttk.Label(input_frame, text="No image loaded", background="lightgray")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=5)

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            button_frame,
            text="▶️ Process",
            command=self.process_input,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_logs
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def setup_right_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        logs_frame = ttk.Frame(notebook)
        progress_frame = ttk.Frame(notebook)
        results_frame = ttk.Frame(notebook)
        debug_frame = ttk.Frame(notebook)

        notebook.add(logs_frame, text="📝 Logs")
        notebook.add(progress_frame, text="📊 Progress")
        notebook.add(results_frame, text="🎯 Results")
        notebook.add(debug_frame, text="🔍 Debug")

        ttk.Label(logs_frame, text="Process Logs:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=30, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(progress_frame, text="Processing Status:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)

        self.status_label = ttk.Label(progress_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.phase_info_text = scrolledtext.ScrolledText(progress_frame, height=25, wrap=tk.WORD)
        self.phase_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(results_frame, text="Analysis Results:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=25, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            export_frame,
            text="💾 Save Phase",
            command=self.save_phase
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            export_frame,
            text="📦 Save Full Cycle",
            command=self.save_cycle
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(debug_frame, text="Debug Information:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        self.debug_text = scrolledtext.ScrolledText(debug_frame, height=30, wrap=tk.WORD)
        self.debug_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def add_log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def save_api_key(self):
        api_key = self.api_key_entry.get()
        self.config['api_key'] = api_key
        save_config(self.config)
        self.api_key = api_key
        messagebox.showinfo("Success", "API key saved successfully!")
        self.add_log("API key updated", "INFO")

    def select_input(self):
        input_type = self.input_type.get()

        try:
            if input_type == "image":
                file_path = filedialog.askopenfilename(
                    title="Select Image",
                    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
                )
                if file_path:
                    self.current_frame = process_image_input(file_path)
                    self.display_preview(self.current_frame)
                    self.add_log(f"Image loaded: {file_path}", "INFO")

            elif input_type == "video":
                file_path = filedialog.askopenfilename(
                    title="Select Video",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov")]
                )
                if file_path:
                    frames = process_video_input(file_path, frame_skip=30)
                    if frames:
                        self.current_frame = frames[0]
                        self.display_preview(self.current_frame)
                        self.add_log(f"Video loaded: {len(frames)} key frames", "INFO")

            elif input_type == "webcam":
                self.current_frame = capture_from_webcam()
                self.display_preview(self.current_frame)
                self.add_log("Webcam capture complete", "INFO")

            elif input_type == "demo":
                self.current_frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
                self.display_preview(self.current_frame)
                self.add_log("Demo mode: synthetic data generated", "INFO")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load input: {str(e)}")
            self.add_log(f"Error loading input: {str(e)}", "ERROR")

    def display_preview(self, frame):
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            img.thumbnail((250, 250))

            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo

    def process_input(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please select an input first!")
            return

        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress!")
            return

        self.processing = True
        self.clear_logs()

        thread = threading.Thread(target=self.run_processing)
        thread.start()

    def run_processing(self):
        try:
            self.add_log("Initializing UAGO system...", "INFO")

            api_key = self.api_key if not self.demo_var.get() else None
            uago = UAGOCore(api_key=api_key, demo_mode=self.demo_var.get())

            def progress_callback(phase, name, progress):
                self.root.after(0, self.update_progress, phase, name, progress)

            self.cycle_data = uago.process_frame(self.current_frame, progress_callback)

            self.root.after(0, self.display_results)
            self.add_log("Processing complete!", "INFO")

        except Exception as e:
            self.add_log(f"Error during processing: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

        finally:
            self.processing = False
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_label.config(text="Complete"))

    def update_progress(self, phase, name, progress):
        self.progress_var.set(progress)
        self.status_label.config(text=f"Phase {phase}: {name}...")

        phase_info = f"\nPhase {phase}: {name}\n"
        phase_info += f"Progress: {progress:.1f}%\n"
        phase_info += "-" * 50 + "\n"

        self.phase_info_text.insert(tk.END, phase_info)
        self.phase_info_text.see(tk.END)

    def display_results(self):
        if not self.cycle_data:
            return

        self.results_text.delete('1.0', tk.END)

        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, "UAGO OBSERVATION CYCLE RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")

        self.results_text.insert(tk.END, f"Timestamp: {self.cycle_data.get('timestamp', 'N/A')}\n")
        self.results_text.insert(tk.END, f"Input Shape: {self.cycle_data.get('input_shape', 'N/A')}\n\n")

        for i in range(1, 8):
            phase_key = f"phase{i}"
            phase_data = self.cycle_data.get("phases", {}).get(phase_key, {})

            phase_config = UAGO_CONFIG['observation_cycle'][i-1]
            self.results_text.insert(tk.END, f"\n{'='*60}\n")
            self.results_text.insert(tk.END, f"PHASE {i}: {phase_config['name']}\n")
            self.results_text.insert(tk.END, f"{'='*60}\n")

            if 'goal' in phase_config:
                self.results_text.insert(tk.END, f"Goal: {phase_config['goal']}\n\n")

            if phase_data:
                formatted = json.dumps(phase_data, indent=2)
                self.results_text.insert(tk.END, formatted + "\n")

        self.results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.results_text.insert(tk.END, "END OF CYCLE\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n")

        self.debug_text.delete('1.0', tk.END)
        debug_json = json.dumps(self.cycle_data, indent=2)
        self.debug_text.insert(tk.END, debug_json)

    def save_phase(self):
        if not self.cycle_data:
            messagebox.showwarning("Warning", "No cycle data to save!")
            return

        phase_num = tk.simpledialog.askinteger("Save Phase", "Enter phase number (1-7):", minvalue=1, maxvalue=7)
        if phase_num:
            phase_key = f"phase{phase_num}"
            phase_data = self.cycle_data.get("phases", {}).get(phase_key, {})

            if phase_data:
                filepath = save_phase_data(phase_num, phase_data)
                if filepath:
                    messagebox.showinfo("Success", f"Phase {phase_num} saved to:\n{filepath}")
                    self.add_log(f"Phase {phase_num} saved to {filepath}", "INFO")
            else:
                messagebox.showwarning("Warning", f"No data for phase {phase_num}")

    def save_cycle(self):
        if not self.cycle_data:
            messagebox.showwarning("Warning", "No cycle data to save!")
            return

        zip_path = save_full_cycle(self.cycle_data)
        if zip_path:
            messagebox.showinfo("Success", f"Full cycle saved to:\n{zip_path}")
            self.add_log(f"Full cycle saved to {zip_path}", "INFO")

    def clear_logs(self):
        self.log_text.delete('1.0', tk.END)
        self.phase_info_text.delete('1.0', tk.END)
        self.progress_var.set(0)
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use('clam')

    app = UAGOApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
