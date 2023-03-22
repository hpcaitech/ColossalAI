from colossalai.device.device_mesh import DeviceMesh


class DeviceMeshManager:
    """
    Device mesh manager is responsible for creating and managing device meshes.
    """

    def __init__(self):
        self.device_mesh_store = dict()

    def create_device_mesh(self, name, *args, **kwargs) -> DeviceMesh:
        """
        Create a device mesh and store it in the manager.

        Args:
            name (str): name of the device mesh
            *args: args for DeviceMesh
            **kwargs: kwargs for DeviceMesh
        """
        # TODO(Yuliang): replace *args, **kwargs with explicit arguments
        if name not in self.device_mesh_store:
            device_mesh = DeviceMesh(*args, **kwargs)
            self.device_mesh_store[name] = device_mesh
            return device_mesh
        else:
            raise ValueError(f'Device mesh {name} already exists.')

    def get(self, name: str) -> DeviceMesh:
        pass

    def destroy(self):
        pass

    def destroy_all(self):
        pass
