import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fileNameList', type=str, default='using_default_location')
args = parser.parse_args()
name_list = args.fileNameList.split(":")

folder_need_check = []
for loc in name_list:
    # Find only the sub-folder of 'example' folder
    if loc.split("/")[0] == "examples":
        # Add the sub-folder into our list
        if loc.split("/")[1] + "/" + loc.split("/")[2] not in folder_need_check:
            folder_need_check.append(loc.split("/")[1] + "/" + loc.split("/")[2])

# Output the result using print. Then the shell can get the values.
for i in folder_need_check:
    print(i, end=' ')
