import subprocess


def execute_cmd(cmd):
    cmd = ' '.join(cmd)
    print(cmd)
    subprocess.call(cmd, shell=True)


def store_tmp(time, bandwidth):
    f = open("tmp.txt", "w")
    f.write(str(time) + ',' + str(bandwidth))
    f.close()


def load_tmp():
    f = open("tmp.txt", "r")
    ln = f.readline().split(",")
    f.close()
    return (float(ln[0]), float(ln[1]))
