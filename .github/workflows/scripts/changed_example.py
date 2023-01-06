import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileNameList', type=str)
    args = parser.parse_args()
    name_list = args.fileNameList.split(":")
    folder_need_check = set()
    for loc in name_list:
        # Find only the sub-folder of 'example' folder
        if loc.split("/")[0] == "examples" and len(loc.split("/")) >= 4:
            folder_need_check.add(loc.split("/")[1] + "/" + loc.split("/")[2])
    # Output the result using print. Then the shell can get the values.
    if len(folder_need_check) == 1 and folder_need_check[0] == '':
        print('no_file_to_check')
    else:
        print(list(folder_need_check))


if __name__ == '__main__':
    main()
