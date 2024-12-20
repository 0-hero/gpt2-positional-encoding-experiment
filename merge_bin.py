import os
import re

def merge_binary_files(output_path, input_prefix):
    """
    Merges multiple binary files with a common prefix and numbered suffix into a single output file.

    :param output_path: Path for the merged output binary file.
    :param input_prefix: Prefix of the input split binary files to merge.
    """
    # Regular expression to match files with the given prefix and numbered suffix (e.g., prefix_001.bin)
    pattern = re.compile(rf"^{re.escape(input_prefix)}_(\d+)\.bin$")

    # List all files in the current directory that match the pattern
    files = [f for f in os.listdir('.') if pattern.match(f)]

    if not files:
        print(f"No files found with prefix '{input_prefix}_' and '.bin' extension to merge.")
        return

    # Sort files based on the numeric part of the filename
    files.sort(key=lambda x: int(pattern.match(x).group(1)))

    print(f"Found {len(files)} parts to merge:")
    for file in files:
        print(f" - {file}")

    with open(output_path, 'wb') as outfile:
        for file in files:
            print(f"Merging {file}...")
            with open(file, 'rb') as infile:
                while True:
                    chunk = infile.read(1024 * 1024)  # Read in 1 MB chunks
                    if not chunk:
                        break
                    outfile.write(chunk)
            print(f"Finished merging {file}.")

    print(f"All parts merged into {output_path}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge split binary files into a single file.")
    parser.add_argument('output_path', help='Path for the merged output binary file.')
    parser.add_argument('input_prefix', help='Prefix of the input split binary files to merge.')

    args = parser.parse_args()

    merge_binary_files(args.output_path, args.input_prefix)