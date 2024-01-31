import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from full_preciction_with_overlap import process_one_file
def select_input_file():
    file_path = filedialog.askopenfilename()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, folder_path)

def run_analysis():
    input_file = input_entry.get()
    output_folder = output_folder_entry.get()
    output_filename = filename_entry.get()

    if not input_file or not output_folder or not output_filename:
        messagebox.showwarning("Warning", "Please select an input file, an output folder, and specify an output filename.")
        return

    # Construct the full output path using the folder and filename
    full_output_path = f"{output_folder}/{output_filename}"

    # Here, you can call your analysis script function and pass the input_file and full_output_path
    # For example: process_image(input_file, full_output_path)
    process_one_file(input_file, full_output_path)
    messagebox.showinfo("Success", "Analysis completed successfully.")

# Set up the main application window
root = tk.Tk()
root.title("BactUnet Analysis Tool")

# Create and place widgets for input file selection
input_entry = tk.Entry(root, width=50)
input_entry.pack()

browse_input_button = tk.Button(root, text="Browse Input File", command=select_input_file)
browse_input_button.pack()

# Create and place widgets for output folder selection
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.pack()

browse_output_button = tk.Button(root, text="Browse Output Folder", command=select_output_folder)
browse_output_button.pack()

# Create and place widgets for output filename entry
filename_entry = tk.Entry(root, width=50)
filename_entry.pack()
filename_entry.insert(0, "output_filename.tif") # You can specify a default filename here

# Create and place the Run Analysis button
run_button = tk.Button(root, text="Run Analysis", command=run_analysis)
run_button.pack()


# Start the GUI event loop
root.mainloop()
