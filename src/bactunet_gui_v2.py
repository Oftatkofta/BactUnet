import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from full_prediction_with_overlap import process_one_file

class BactUnetGUI:
    """
    A class to create a GUI for the BactUnet analysis tool.
    The tool allows users to select an input file, specify an output folder, and run an analysis using a pre-defined function.
    """
    def __init__(self, root):
        """
        Initialize the BactUnetGUI instance.

        Parameters:
        root (tk.Tk): The root window of the tkinter GUI.
        """
        self.root = root
        self.root.title("BactUnet Analysis Tool")
        
        # Create and layout the GUI widgets
        self.create_widgets()

    def create_widgets(self):
        """
        Create and layout the widgets used in the BactUnet analysis tool GUI.
        This includes buttons for selecting input files, specifying output folders, and running the analysis.
        """
        # Input file selection
        self.input_label = tk.Label(self.root, text="Input File:")
        self.input_label.grid(row=0, column=0, sticky="w")
        
        self.input_entry = tk.Entry(self.root, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.input_button = tk.Button(self.root, text="Browse", command=self.select_input_file)
        self.input_button.grid(row=0, column=2, padx=5, pady=5)

        # Output folder selection
        self.output_folder_label = tk.Label(self.root, text="Output Folder:")
        self.output_folder_label.grid(row=1, column=0, sticky="w")
        
        self.output_folder_entry = tk.Entry(self.root, width=50)
        self.output_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.output_folder_button = tk.Button(self.root, text="Browse", command=self.select_output_folder)
        self.output_folder_button.grid(row=1, column=2, padx=5, pady=5)

        # Run analysis button
        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=2, column=1, pady=10)

        # Scrolled text area for logs or results (optional)
        self.log_area = scrolledtext.ScrolledText(self.root, width=60, height=15, wrap=tk.WORD)
        self.log_area.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    def select_input_file(self):
        """
        Opens a file dialog to select an input file and updates the input entry widget.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)

    def select_output_folder(self):
        """
        Opens a folder dialog to select an output directory and updates the output folder entry widget.
        """
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_path)

    def run_analysis(self):
        """
        Runs the analysis on the selected input file and saves results in the output folder.
        Validates the input file and output folder before running the analysis.
        If an error occurs during the process, an error message is displayed, and the error is logged in the log area.
        """
        input_file = self.input_entry.get()
        output_folder = self.output_folder_entry.get()

        # Validate input and output paths
        if not input_file or not output_folder:
            messagebox.showerror("Error", "Please provide valid input file and output folder.")
            return

        try:
            # Run the analysis function
            process_one_file(input_file, output_folder)
            messagebox.showinfo("Success", "Analysis completed successfully.")
        except Exception as e:
            # Handle any exceptions that occur during the analysis
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.log_area.insert(tk.END, f"Error: {str(e)}\n")
            self.log_area.see(tk.END)

if __name__ == "__main__":
    # Create the root window and run the BactUnetGUI application
    root = tk.Tk()
    app = BactUnetGUI(root)
    root.mainloop()