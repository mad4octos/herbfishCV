import os
import sys

def renumber_images(folder):
    # Get all jpg files
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    
    if not files:
        print("No JPG files found.")
        return

    # Sort files by name
    files.sort()

    # Check if first file already 0001.jpg
    if files[0] == "0001.jpg":
        print("First file already 0001.jpg — no renaming needed.")
        return

    print("Renaming files to start at 0001.jpg...")

    # Step 1: rename to temporary names to avoid collisions
    temp_files = []
    for i, filename in enumerate(files):
        old_path = os.path.join(folder, filename)
        temp_name = f"__temp_{i}.jpg"
        temp_path = os.path.join(folder, temp_name)

        os.rename(old_path, temp_path)
        temp_files.append(temp_name)

    # Step 2: rename to final sequence
    for i, filename in enumerate(temp_files, start=1):
        old_path = os.path.join(folder, filename)
        new_name = f"{i:04d}.jpg"
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} files starting from 0001.jpg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python renumber_images.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    renumber_images(folder_path)