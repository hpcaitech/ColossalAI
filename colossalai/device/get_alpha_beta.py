import torch.multiprocessing as mp

from .get_one_alpha_beta import get_one_alpha_beta
from .utils import execute_cmd, load_tmp

PROFILE = "profile.csv"


def list_to_string(ilist, knob=","):
    ostr = ""
    l = len(ilist)
    assert l > 0
    for (k, v) in enumerate(ilist):
        ostr += str(v)
        if k < l - 1:
            ostr += knob
    return ostr


def store_profile(ilist, filename):
    ostr = list_to_string(ilist) + "\n"
    f = open(filename, "a+")
    f.writelines(ostr)
    f.close()


def profile_all_alpha_beta(width, height, physical_devices):    # Assumes only 4 physical devices
    num_devices = 4
    logical_meshes = [(4, 1), (2, 2), (1, 4)]
    assert (width, height) in logical_meshes
    physical_meshes = [physical_devices]
    physical_alpha_beta = []
    if (width, height) == (2, 2):
        physical_meshes = []
        for i in range(num_devices - 1):
            for j in range(i + 1, num_devices):
                physical_meshes.append([i, j])
    for device_list in physical_meshes:
        mp.spawn(get_one_alpha_beta, nprocs=len(device_list))
        (a, b) = load_tmp()
        physical_alpha_beta.append((a, b))
        store_profile([list_to_string(device_list, "-"), a, b], PROFILE)
    return (physical_meshes, physical_alpha_beta)


def get_alpha_beta(width, height, physical_devices):
    profile_all_alpha_beta(width, height, physical_devices)
    execute_cmd(["rm", "tmp.txt"])
