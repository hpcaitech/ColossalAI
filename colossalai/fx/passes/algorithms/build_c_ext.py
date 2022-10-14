from setuptools import setup, Extension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = [Extension(
    'dynamic_programs_C_version',
    sources=[os.path.join(this_dir, 'dynamic_programs.c')],
)]

setup(
    name='rotor c extension',
    version='0.1',
    description='rotor c extension for faster dp computing',
    ext_modules=ext_modules,
)
