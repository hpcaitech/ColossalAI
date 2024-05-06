from colossalai.cluster.device_mesh_manager import DeviceMeshInfo, DeviceMeshManager
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import spawn


def check_device_mesh_manager(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    device_mesh_manager = DeviceMeshManager()
    # TODO(ver217): this test is strictly relies on hardware, temporary skip it
    # device_mesh_info_auto = DeviceMeshInfo(physical_ids=[0, 1, 2, 3],)
    # device_mesh_auto = device_mesh_manager.create_device_mesh('0', device_mesh_info_auto)
    # assert device_mesh_auto.shape == (2, 2)
    # assert device_mesh_auto._logical_mesh_id.tolist() == [[0, 1], [2, 3]]

    device_mesh_info_with_shape = DeviceMeshInfo(
        physical_ids=[0, 1, 2, 3],
        mesh_shape=(2, 2),
    )
    device_mesh_with_shape = device_mesh_manager.create_device_mesh("1", device_mesh_info_with_shape)

    assert device_mesh_with_shape.shape == (2, 2)
    assert device_mesh_with_shape._logical_mesh_id.tolist() == [[0, 1], [2, 3]]


def test_device_mesh_manager():
    spawn(check_device_mesh_manager, 4)


if __name__ == "__main__":
    test_device_mesh_manager()
