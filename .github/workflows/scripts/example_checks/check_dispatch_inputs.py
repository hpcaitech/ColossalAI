import argparse
import os


def check_inputs(input_list):
    for path in input_list:
        real_path = os.path.join("examples", path)
        if not os.path.exists(real_path):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fileNameList", type=str, help="List of file names")
    args = parser.parse_args()
    name_list = args.fileNameList.split(",")
    is_correct = check_inputs(name_list)

    if is_correct:
        print("success")
    else:
        print("failure")


if __name__ == "__main__":
    main()
