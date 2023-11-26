import os


def show_files(path, all_files):
    # Traverse all the folder/file in current directory
    file_list = os.listdir(path)
    # Determine the element is folder or file. If file, pass it into list, if folder, recurse.
    for file_name in file_list:
        # Get the abs directory using os.path.join() and store into cur_path.
        cur_path = os.path.join(path, file_name)
        # Determine whether folder
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            all_files.append(cur_path)
    return all_files


def join(input_list, sep=None):
    return (sep or " ").join(input_list)


def main():
    contents = show_files("examples/", [])
    all_loc = []
    for file_loc in contents:
        split_loc = file_loc.split("/")
        # must have two sub-folder levels after examples folder, such as examples/images/vit is acceptable, examples/images/README.md is not, examples/requirements.txt is not.
        if len(split_loc) >= 4:
            re_loc = "/".join(split_loc[1:3])
            if re_loc not in all_loc:
                all_loc.append(re_loc)
    print(all_loc)


if __name__ == "__main__":
    main()
