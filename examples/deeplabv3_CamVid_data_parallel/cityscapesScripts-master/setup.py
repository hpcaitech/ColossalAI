#!/usr/bin/env python
#
# Enable cython support for slightly faster eval scripts:
# python -m pip install cython numpy
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup, find_packages

include_dirs = []
ext_modules = []
if 'CYTHONIZE_EVAL' in os.environ:
    from Cython.Build import cythonize
    import numpy as np
    include_dirs = [np.get_include()]

    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

    pyxFile = os.path.join("cityscapesscripts",
                           "evaluation", "addToConfusionMatrix.pyx")
    ext_modules = cythonize(pyxFile)

with open("README.md") as f:
    readme = f.read()

with open(os.path.join('cityscapesscripts', 'VERSION')) as f:
    version = f.read().strip()

console_scripts = [
    'csEvalPixelLevelSemanticLabeling = cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling:main',
    'csEvalInstanceLevelSemanticLabeling = cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling:main',
    'csEvalPanopticSemanticLabeling = cityscapesscripts.evaluation.evalPanopticSemanticLabeling:main',
    'csEvalObjectDetection3d = cityscapesscripts.evaluation.evalObjectDetection3d:main',
    'csCreateTrainIdLabelImgs = cityscapesscripts.preparation.createTrainIdLabelImgs:main',
    'csCreateTrainIdInstanceImgs = cityscapesscripts.preparation.createTrainIdInstanceImgs:main',
    'csCreatePanopticImgs = cityscapesscripts.preparation.createPanopticImgs:main',
    'csDownload = cityscapesscripts.download.downloader:main',
    'csPlot3dDetectionResults = cityscapesscripts.evaluation.plot3dResults:main'
]

gui_scripts = [
    'csViewer = cityscapesscripts.viewer.cityscapesViewer:main [gui]',
    'csLabelTool = cityscapesscripts.annotation.cityscapesLabelTool:main [gui]'
]

config = {
    'name': 'cityscapesScripts',
    'description': 'Scripts for the Cityscapes Dataset',
    'long_description': readme,
    'long_description_content_type': "text/markdown",
    'author': 'Marius Cordts',
    'url': 'https://github.com/mcordts/cityscapesScripts',
    'author_email': 'mail@cityscapes-dataset.net',
    'license': 'https://github.com/mcordts/cityscapesScripts/blob/master/license.txt',
    'version': version,
    'install_requires': ['numpy', 'matplotlib', 'pillow', 'appdirs', 'pyquaternion', 'coloredlogs', 'tqdm', 'typing'],
    'setup_requires': ['setuptools>=18.0'],
    'extras_require': {
        'gui': ['PyQt5']
    },
    'packages': find_packages(),
    'scripts': [],
    'entry_points': {'gui_scripts': gui_scripts,
                     'console_scripts': console_scripts},
    'package_data': {'': ['VERSION', 'icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': include_dirs
}

setup(**config)
