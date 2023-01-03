import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileNameList', type=str, default='using_default_location')
    args = parser.parse_args()
    name_list = args.fileNameList.split(":")
    folder_need_check = []
    for loc in name_list:
        # Avoid '' sometimes caused by changedFile action.
        if loc != '' and loc != 'README.md':
            folder_need_check.append(loc)
    print(folder_need_check)


if __name__ == '__main__':
    main()