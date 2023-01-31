from setuptools import find_packages, setup

setup(
    name='siu',
    packages=find_packages(exclude=(
        'playground',
        'tests',
        '*.egg-info',
    )),
    version='0.0.1',
    description='Solver integration utils for Colossal-AI',
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
)
