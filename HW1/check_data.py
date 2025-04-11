import os
from collections import defaultdict


def count_files_in_folders(root_dir):
    # Dictionary to store folder counts
    folder_counts = defaultdict(int)

    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get the immediate parent folder name
        folder_name = os.path.basename(dirpath)
        # Count all files in this directory
        num_files = len(filenames)
        if num_files > 0:  # Only count folders that contain files
            folder_counts[folder_name] = num_files

    return folder_counts


def main():
    # Replace this with your dataset directory path
    dataset_path = "./flower/train"  # Adjust this path as needed

    if not os.path.exists(dataset_path):
        print(f"Error: Directory {dataset_path} does not exist!")
        return

    # Get the counts
    counts = count_files_in_folders(dataset_path)

    # Print results
    print("Number of files in each folder:")
    print("-" * 30)
    for folder, count in sorted(counts.items()):
        print(f"{folder}: {count} files")


if __name__ == "__main__":
    main()
