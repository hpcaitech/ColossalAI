import argparse

# 初始化参数构造器
parser = argparse.ArgumentParser()
# print('11')
# 在参数构造器中添加两个命令行参数
parser.add_argument('--fileNameList', type=str, default='Siri12=')
# parser.add_argument('--message', type=str, default=',Welcom to Python World!')

# 获取所有的命令行参数
args = parser.parse_args()
# print('22')
# end_name = str(args.filename).split(".")[-1]
#
# print('Hi ' + str(args.fileNameList) + str(args.message))

name_list = args.fileNameList.split("&&&@@@")
# print(name_list, 'name_list')
folder_need_check = []
for loc in name_list:
    if loc.split("/")[0] == "examples":
        if loc.split("/")[1] + "/" + loc.split("/")[2] not in folder_need_check:
            folder_need_check.append(loc.split("/")[1] + "/" + loc.split("/")[2])


for i in folder_need_check:
    print(i, end=':')