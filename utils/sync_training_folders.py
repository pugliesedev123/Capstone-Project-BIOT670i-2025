#!/usr/bin/env python3

import os
import argparse
import shutil


def get_subfolders(path):
    """Return a set of immediate subfolder names within a given path."""
    return {
        entry.name for entry in os.scandir(path)
        if entry.is_dir()
    }


def compare_folders(template_dir, target_dir, sync=False, wipe=False):
    template_folders = get_subfolders(template_dir)
    target_folders = get_subfolders(target_dir)

    missing = template_folders - target_folders
    extra = target_folders - template_folders

    folder_name = os.path.basename(target_dir)

    if missing:
        print(f"[{folder_name}] Missing folders (in template but not here):")
        for folder in sorted(missing):
            print(f"  - {folder}")
            if sync:
                os.makedirs(os.path.join(target_dir, folder), exist_ok=True)

    if extra:
        print(f"[{folder_name}] Extra folders (not in template):")
        for folder in sorted(extra):
            print(f"  + {folder}")
            if wipe:
                to_delete = os.path.join(target_dir, folder)
                shutil.rmtree(to_delete)
                print(f"    Removed {folder}")

    if not missing and not extra:
        print(f"[{folder_name}] All folders match the template.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare and optionally sync or clean owner-* folders to match owner-template."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Create missing folders in each owner-* directory to match owner-template."
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete extra folders from each owner-* directory if they are not in owner-template."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.path.join("data", "train"),
        help="Path to the training directory (default: data/train)"
    )
    args = parser.parse_args()

    template_dir = os.path.join(args.train_dir, "owner-template")

    if not os.path.isdir(template_dir):
        print(f"Error: Template directory not found at {template_dir}")
        return

    for entry in os.scandir(args.train_dir):
        if entry.is_dir() and entry.name.startswith("owner-") and entry.name != "owner-template" and entry.name != "owner-combined":
            compare_folders(template_dir, entry.path, sync=args.sync, wipe=args.wipe)

if __name__ == "__main__":
    main()
