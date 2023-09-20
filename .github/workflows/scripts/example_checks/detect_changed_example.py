import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fileNameList", type=str, help="The list of changed files")
    args = parser.parse_args()
    name_list = args.fileNameList.split(":")
    folder_need_check = set()
    for loc in name_list:
        # Find only the sub-sub-folder of 'example' folder
        # the examples folder structure is like
        # - examples
        #   - area
        #     - application
        #       - file
        if loc.split("/")[0] == "examples" and len(loc.split("/")) >= 4:
            folder_need_check.add("/".join(loc.split("/")[1:3]))
    # Output the result using print. Then the shell can get the values.
    print(list(folder_need_check))


if __name__ == "__main__":
    main()
