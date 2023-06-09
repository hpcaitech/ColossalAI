from colossalai.device.device_mesh import DeviceMesh
import torch


def test_device_mesh():
    physical_mesh_id = torch.arange(0, 16).reshape(2, 8)
    mesh_shape = (4, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10,11],
    #  [12,13,14,15]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    assert device_mesh.convert_map[5] == [1, 1]
    assert device_mesh.convert_map[11] == [2, 3]
    assert device_mesh.global_rank_to_process_groups_with_logical_rank(0)[0] == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert device_mesh.global_rank_to_process_groups_with_logical_rank(2)[1] == [[0, 0], [0, 1], [0, 2], [0, 3]]
    assert device_mesh.global_rank_to_process_groups_with_global_rank(2)[1] == [0, 1, 2, 3]


if __name__ == '__main__':
    test_device_mesh()
