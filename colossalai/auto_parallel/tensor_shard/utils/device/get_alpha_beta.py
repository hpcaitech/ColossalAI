from colossalai.device import execute_cmd, load_tmp

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


def profile_all_alpha_beta(width, height):    # Assumes only 4 physical devices
    num_devices = 4
    physical_devices = [0, 1, 2, 3]
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
        omp_prefix = "OMP_NUM_THREADS=1"
        cuda_prefix = "CUDA_VISIBLE_DEVICES=" + list_to_string(device_list)
        python_prefix = f"python -m torch.distributed.run --nproc_per_node={len(device_list)} --master_port 11000 get_one_alpha_beta.py"
        execute_cmd([omp_prefix, cuda_prefix, python_prefix])
        (a, b) = load_tmp()
        physical_alpha_beta.append((a, b))
        store_profile([list_to_string(device_list, "-"), a, b], PROFILE)
    return (physical_meshes, physical_alpha_beta)


def get_alpha_beta(width, height):

    profile_all_alpha_beta(width, height)


if __name__ == "__main__":
    get_alpha_beta(2, 2)
    execute_cmd(["rm", "tmp.txt"])
