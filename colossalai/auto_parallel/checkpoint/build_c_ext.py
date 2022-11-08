import os

from setuptools import Extension, setup

this_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = [Extension(
    'rotorc',
    sources=[os.path.join(this_dir, 'ckpt_solver_rotor.c')],
)]

setup(
    name='rotor c extension',
    version='0.1',
    description='rotor c extension for faster dp computing',
    ext_modules=ext_modules,
)
