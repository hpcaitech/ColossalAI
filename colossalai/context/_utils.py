import math


def set_parallel_size(obj, config: dict, key: str, attr_name: str):
    if key in config:
        ele = config[key]
        if isinstance(ele, int):
            setattr(obj, attr_name, ele)
        elif isinstance(ele, dict):
            setattr(obj, attr_name, ele['size'])
        else:
            raise NotImplementedError(
                f"Parallel configuration does not support this kind of argument, please use int or dict"
            )


def add_tensor_pg(pg_init, mode, size, depth=None):
    if mode == '1d':
        pg_init.append(dict(
            type='Initializer1D',
            parallel_size=size
        ))
    elif mode == '2d':
        dim = math.floor(math.sqrt(size))
        pg_init.append(dict(
            type='Initializer2D_Col',
            summa_dim=dim
        ))
        pg_init.append(dict(
            type='Initializer2D_Row',
            summa_dim=dim
        ))
    elif mode == '2.5d':
        dim = math.floor(math.sqrt(size // depth))
        pg_init.append(dict(
            type='Initializer_Tesseract_ROW',
            tesseract_dim=dim,
            tesseract_dep=depth
        ))
        pg_init.append(dict(
            type='Initializer_Tesseract_COL',
            tesseract_dim=dim,
            tesseract_dep=depth
        ))
        pg_init.append(dict(
            type='Initializer_Tesseract_DEP',
            tesseract_dim=dim,
            tesseract_dep=depth
        ))
        pg_init.append(dict(
            type='Initializer_Tesseract_XZ',
            tesseract_dim=dim,
            tesseract_dep=depth
        ))
    elif mode == '3d':
        dim = math.floor(math.pow(size, 1.0 / 3.0) + 0.5)
        pg_init.append(dict(
            type='ParallelInitializer3D_Input',
            depth=dim
        ))
        pg_init.append(dict(
            type='ParallelInitializer3D_Weight',
            depth=dim
        ))
        pg_init.append(dict(
            type='ParallelInitializer3D_Output',
            depth=dim
        ))
    else:
        raise NotImplementedError("This kind of tensor splitting has not been implemented yet")
