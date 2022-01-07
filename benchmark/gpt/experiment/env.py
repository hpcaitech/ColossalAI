import os

__all__ = [
    'set_env',
]


def append_path(env, path):
    if env in os.environ:
        os.environ[env] = f'{os.environ[env]}:{path}'
    else:
        os.environ[env] = path


def set_env():
    append_path(
        'CPATH', '/project/home/p200012/shenggan-c/miniconda3/pkgs/libaio-0.3.111-h14c3975_0/include')
    append_path('LIBRARY_PATH',
                '/project/home/p200012/shenggan-c/miniconda3/pkgs/libaio-0.3.111-h14c3975_0/lib')
    append_path('LD_LIBRARY_PATH',
                '/project/home/p200012/shenggan-c/miniconda3/pkgs/libaio-0.3.111-h14c3975_0/lib')
