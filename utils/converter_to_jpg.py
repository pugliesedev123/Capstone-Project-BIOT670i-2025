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

                    # JPEG doesn't support alpha; convert RGBA/LA/P to RGB first.
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")

                    image.save(jpg_path, format="JPEG", quality=95)
                    os.remove(heic_path)

                    print(f"Converted and deleted: {heic_path}")
                    count += 1

                except Exception as e:
                    print(f"Failed to convert {heic_path}: {e}")
                    
    print(f"\nDone. Converted {count} HEIC file(s).")


def convert_pngs_to_jpegs(root_dir):
    count = 0

    # Walk directories looking for .png.
    for dirpath, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                png_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(png_path)[0] + ".jpg"

                # Convert .png to JPEG if possible using Pillow,
                # delete old photo afterwards.
                try:
                    image = Image.open(png_path)

                    # JPEG doesn't support alpha; convert RGBA/LA/P to RGB first.
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")

                    image.save(jpg_path, format="JPEG", quality=95)
                    os.remove(png_path)

                    print(f"Converted and deleted: {png_path}")
                    count += 1

                except Exception as e:
                    print(f"Failed to convert {png_path}: {e}")

    print(f"\nDone. Converted {count} PNG file(s).")


def convert_gifs_to_jpegs(root_dir):
    count = 0

    # Walk directories looking for .gif.
    for dirpath, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".gif"):
                gif_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(gif_path)[0] + ".jpg"

                # Convert .gif to JPEG if possible using Pillow,
                # delete old photo afterwards.
                # If the GIF is animated, use the first frame.
                try:
                    image = Image.open(gif_path)

                    # Ensure we're on the first frame (works for animated + non-animated GIFs).
                    try:
                        image.seek(0)
                    except Exception:
                        pass

                    # GIFs are often palette-based; convert to RGB for JPEG.
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")
                    elif image.mode != "RGB":
                        image = image.convert("RGB")

                    image.save(jpg_path, format="JPEG", quality=95)
                    os.remove(gif_path)

                    print(f"Converted and deleted: {gif_path}")
                    count += 1

                except Exception as e:
                    print(f"Failed to convert {gif_path}: {e}")

    print(f"\nDone. Converted {count} GIF file(s).")


def main():

    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Convert HEIC and/or PNG and/or GIF files to JPEG in the target directory and delete originals.")
    parser.add_argument("--target-dir", type=str, default=os.path.join("data", "train"), help="Path to the target directory (default: data/train)")
    parser.add_argument("--file-type", type=str, choices=["heic", "png", "gif", "all"], default="heic", help="Choose which file type(s) to convert: heic, png, gif, or all (heic+png+gif) (default: heic)")

    args = parser.parse_args()

    if not os.path.isdir(args.target_dir):
        print(f"Error: Directory not found at {args.target_dir}")
        return

    if args.file_type == "heic":
        convert_heics_to_jpegs(args.target_dir)

    elif args.file_type == "png":
        convert_pngs_to_jpegs(args.target_dir)

    elif args.file_type == "gif":
        convert_gifs_to_jpegs(args.target_dir)

    elif args.file_type == "all":
        convert_heics_to_jpegs(args.target_dir)
        convert_pngs_to_jpegs(args.target_dir)
        convert_gifs_to_jpegs(args.target_dir)


if __name__ == "__main__":
    main()