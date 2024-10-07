import os
import sys
import shutil

def main():
    """
    Main function to rename files in a specified input directory and save them to an output directory.
    Each file in the input directory is renamed by appending '_masks' to its original name before the file extension.

    Command line arguments:
    - input_directory: The directory containing the files to be renamed.
    - output_directory: The directory where renamed files will be saved.
    """
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <input_directory> <output_directory>")
        sys.exit(1)

    indir = sys.argv[1]
    outdir = sys.argv[2]

    # Check if input directory exists
    if not os.path.exists(indir):
        print(f"Error: Input directory '{indir}' does not exist.")
        sys.exit(1)

    # Create output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # List files in input directory
    files = os.listdir(indir)
    
    if not files:
        print(f"No files found in the input directory '{indir}'.")
        sys.exit(1)

    # Process each file
    for f in files:
        # Construct old and new file paths
        oldname = os.path.join(indir, f)
        filename, ext = os.path.splitext(f)
        newname = os.path.join(outdir, f"{filename}_masks{ext}")

        # Rename the file with error handling
        try:
            shutil.move(oldname, newname)
            print(f"Renamed: '{oldname}' -> '{newname}'")
        except Exception as e:
            print(f"Error renaming '{oldname}' to '{newname}': {e}")

    print("Renaming process completed successfully.")

if __name__ == "__main__":
    main()