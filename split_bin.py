import os

def split_binary_file(input_path, output_prefix, max_size=50 * 1024**3):
    """
    Splits a large binary file into smaller files each less than max_size bytes.

    :param input_path: Path to the input binary file.
    :param output_prefix: Prefix for the output split files.
    :param max_size: Maximum size in bytes for each split file (default is 50 GB).
    """
    file_size = os.path.getsize(input_path)
    num_parts = (file_size + max_size - 1) // max_size  # Ceiling division
    num_digits = len(str(num_parts)) if num_parts > 0 else 1  # Determine number of digits for padding

    with open(input_path, 'rb') as infile:
        for part in range(1, num_parts + 1):
            part_filename = f"{output_prefix}_{str(part).zfill(3)}.bin"
            with open(part_filename, 'wb') as outfile:
                bytes_written = 0
                while bytes_written < max_size:
                    # Read in chunks of 1 MB
                    chunk_size = min(1024 * 1024, max_size - bytes_written)
                    data = infile.read(chunk_size)
                    if not data:
                        break
                    outfile.write(data)
                    bytes_written += len(data)
            print(f"Created {part_filename} with {bytes_written} bytes.")

    print("File splitting complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a large binary file into smaller parts.")
    parser.add_argument('input_path', help='Path to the input binary file.')
    parser.add_argument('output_prefix', help='Prefix for the output split files.')
    parser.add_argument('--max_size_gb', type=float, default=50.0, help='Maximum size per split file in GB (default: 50 GB).')

    args = parser.parse_args()
    max_size_bytes = int(args.max_size_gb * 1024**3)

    split_binary_file(args.input_path, args.output_prefix, max_size=max_size_bytes)