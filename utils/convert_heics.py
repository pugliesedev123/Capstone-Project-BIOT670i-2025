#!/usr/bin/env python3

import os
import argparse
from PIL import Image
import pillow_heif


def convert_heics_to_jpegs(root_dir):
    count = 0

    # Walk directories looking for .heic.
    for dirpath, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".heic"):
                heic_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(heic_path)[0] + ".jpg"

                # Convert .heic to JPEG if possible using pillow_heif,
                # delete old photo afterwards.
                try:
                    heif_file = pillow_heif.read_heif(heic_path)
                    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                    image.save(jpg_path, format="JPEG", quality=95)
                    os.remove(heic_path)

                    print(f"Converted and deleted: {heic_path}")
                    count += 1

                except Exception as e:
                    print(f"Failed to convert {heic_path}: {e}")
                    
    print(f"\nDone. Converted {count} HEIC file(s).")


def main():

    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Convert all HEIC files to JPEG in the target directory and delete originals.")
    parser.add_argument("--target-dir", type=str, default=os.path.join("data", "train"), help="Path to the target directory (default: data/train)")

    args = parser.parse_args()

    if not os.path.isdir(args.target_dir):
        print(f"Error: Directory not found at {args.target_dir}")
        return

    convert_heics_to_jpegs(args.target_dir)


if __name__ == "__main__":
    main()
