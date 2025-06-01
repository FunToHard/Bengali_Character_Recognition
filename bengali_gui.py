import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import platform
from PIL import Image, ImageTk
import torch
from recognize_text import load_model, recognize_text, get_bengali_char_map
import os

class BengaliRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bengali Character Recognition")
        self.root.geometry("800x600")
        
        # Setup model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = load_model("models/checkpoint_epoch_49.pth", self.device)
        self.char_map = get_bengali_char_map()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create and configure drop zone
        self.drop_frame = ttk.LabelFrame(self.main_frame, text="Drop Image Here", padding="10")
        self.drop_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.drop_frame.grid_columnconfigure(0, weight=1)
        self.drop_frame.grid_rowconfigure(0, weight=1)
        
        # Image preview label
        self.image_label = ttk.Label(self.drop_frame, text="Drag and drop image here\nor click to browse")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results text area
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Recognition Results", padding="5")
        self.results_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=10, width=60)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Bind events
        self.image_label.bind("<Button-1>", self.browse_file)
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.handle_drop)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.status_var.set("Ready")

    def browse_file(self, event=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.process_image(file_path)
            
    def handle_drop(self, event):
        # Handle dropped files
        file_path = event.data
        
        # Clean up file path based on OS
        if platform.system() == "Windows":
            # Remove curly braces and quotes if present
            file_path = file_path.strip("{}").strip('"')
        else:
            file_path = file_path.strip()
        
        # Check if it"s an image file
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.process_image(file_path)
        else:
            self.status_var.set("Error: Please drop an image file")

    def process_image(self, file_path):
        try:
            # Update status
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Display image preview
            image = Image.open(file_path)
            # Resize image to fit in the window while maintaining aspect ratio
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference!
            
            # Recognize text
            results = recognize_text(self.model, file_path, self.device, self.char_map)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            text = ""
            self.results_text.insert(tk.END, "Recognition Results:\n\n")
            for char, confidence in results:
                text += char
                result_line = f"Character: {char}, Confidence: {confidence*100:.2f}%\n"
                self.results_text.insert(tk.END, result_line)
            
            self.results_text.insert(tk.END, f"\nComplete text: {text}")
            self.status_var.set("Recognition complete")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error processing image:\n{str(e)}")

def main():
    root = TkinterDnD.Tk()  # Use TkinterDnD instead of regular Tk
    
    # Set window icon and style
    style = ttk.Style()
    style.theme_use("clam")  # Use "clam" theme for better looking widgets
    
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Configure style
    style.configure("TLabelframe", padding=10)
    style.configure("TButton", padding=5)
    
    # Create and run the application
    app = BengaliRecognizerGUI(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry("{}x{}+{}+{}".format(width, height, x, y))
    
    root.mainloop()

if __name__ == "__main__":
    main()
