import argparse
import os


def compare_dirs(dir1, dir2):
    # First, we need to check if the two directories exist
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return False

    # Now, we compare the list of items in each directory
    items1 = os.listdir(dir1)
    items2 = os.listdir(dir2)

    # If the number of items in each directory is different, the directories are different
    if len(items1) != len(items2):
        return False

    # For each item in the first directory, we check if there is a corresponding item in the second directory
    for item in items1:
        item_path1 = os.path.join(dir1, item)
        item_path2 = os.path.join(dir2, item)

        # If the corresponding item doesn't exist in the second directory, the directories are different
        if not os.path.exists(item_path2):
            print(f"Found mismatch: {item_path1}, {item_path2}")
            return False

        # If the corresponding item is a directory, we compare the two directories recursively
        if os.path.isdir(item_path1) and os.path.isdir(item_path2):
            if not compare_dirs(item_path1, item_path2):
                print(f"Found mismatch: {item_path1}, {item_path2}")
                return False

        # both are files
        elif os.path.isfile(item_path1) and os.path.isfile(item_path2):
            continue

        # If the corresponding item is not a file or a directory, the directories are different
        else:
            print(f"Found mismatch: {item_path1}, {item_path2}")
            return False

    # If all items are the same, the directories are the same
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="The directory where the multi-language source files are kept.")
    args = parser.parse_args()

    i18n_folders = os.listdir(args.directory)
    i18n_folders = [os.path.join(args.directory, val) for val in i18n_folders]

    if len(i18n_folders) > 1:
        for i in range(1, len(i18n_folders)):
            dir1 = i18n_folders[0]
            dir2 = i18n_folders[i]
            print(f"comparing {dir1} vs {dir2}")
            match = compare_dirs(i18n_folders[0], i18n_folders[i])

            if not match:
                print(
                    f"{dir1} and {dir2} don't match, please ensure that your documentation is available in different languages"
                )
            else:
                print(f"{dir1} and {dir2} match")
