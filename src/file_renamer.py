import os
import shutil
import argparse

def main():
    """
    Main function to rename files in a specified input directory and save them to an output directory.
    Each file in the input directory is renamed by appending '_masks' to its original name before the file extension.

    Command line arguments:
    - input_directory: The directory containing the files to be renamed.
    - output_directory: The directory where renamed files will be saved.
    """
    # Set up argument parsing for input and output directories
    parser = argparse.ArgumentParser(description="Rename files in a directory by appending '_masks' to their original names.")
    parser.add_argument("input_directory", help="The directory containing the files to be renamed.")
    parser.add_argument("output_directory", help="The directory where renamed files will be saved.")
    args = parser.parse_args()

    indir = args.input_directory
    outdir = args.output_directory

    # Check if input directory exists
    if not os.path.exists(indir):
        print(f"Error: Input directory '{indir}' does not exist.")
        sys.exit(1)

    # Create output directory if it does not exist
    os.makedirs(outdir, exist_ok=True)

    # List files in input directory
    files = [f for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f))]
    
    # If no files are found, exit with an appropriate message
    if not files:
        print(f"No files found in the input directory '{indir}'.")
        sys.exit(1)

    # Track the number of files successfully renamed
    renamed_count = 0

    # Process each file in the input directory
    for f in files:
        # Construct old and new file paths
        oldname = os.path.join(indir, f)
        filename, ext = os.path.splitext(f)
        newname = os.path.join(outdir, f"{filename}_masks{ext}")

        # Check if the destination file already exists
        if os.path.exists(newname):
            # Prompt user for action if the destination file already exists
            user_input = input(f"File '{newname}' already exists. Overwrite? (y/n): ")
            if user_input.lower() != 'y':
                print(f"Skipping file '{oldname}'.")
                continue

        # Rename the file with error handling
        try:
            # Move the file to the new location with the updated name
            shutil.move(oldname, newname)
            print(f"Renamed: '{oldname}' -> '{newname}'")
            renamed_count += 1
        except FileNotFoundError:
            # Handle case where the file to be renamed is not found
            print(f"Error: File '{oldname}' was not found.")
        except PermissionError:
            # Handle case where permission is denied for the file operation
            print(f"Permission denied when trying to rename '{oldname}'.")
        except OSError as e:
            # Handle other OS-related errors
            print(f"OS error while renaming '{oldname}' to '{newname}': {e}")

    # Print a summary of the renaming process
    print(f"Renaming process completed successfully. {renamed_count} files were renamed.")

# Entry point of the script
if __name__ == "__main__":
    main()