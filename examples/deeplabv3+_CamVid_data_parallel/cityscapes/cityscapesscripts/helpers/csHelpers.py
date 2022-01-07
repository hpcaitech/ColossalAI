#!/usr/bin/python
#
# Various helper methods and includes for Cityscapes
#

# Python imports
from __future__ import print_function, absolute_import, division
import os
import sys
import getopt
import glob
import math
import json
from collections import namedtuple
import logging
import traceback

# Image processing
from PIL import Image
from PIL import ImageDraw

# Numpy for datastructures
import numpy as np

# Cityscapes modules
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels import labels, name2label, id2label, trainId2label, category2labels


def printError(message):
    """Print an error message and quit"""
    print('ERROR: ' + str(message))
    sys.exit(-1)


class colors:
    """Class for colors"""
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def getColorEntry(val, args):
    """Colored value output if colorized flag is activated."""

    if not args.colorized:
        return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN


# Cityscapes files have a typical filename structure
# <city>_<sequenceNb>_<frameNb>_<type>[_<type2>].<ext>
# This class contains the individual elements as members
# For the sequence and frame number, the strings are returned, including leading zeros
CsFile = namedtuple('csFile', ['city', 'sequenceNb', 'frameNb', 'type', 'type2', 'ext'])


def getCsFileInfo(fileName):
    """Returns a CsFile object filled from the info in the given filename"""
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError('Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName))
    if len(parts) == 5:
        csFile = CsFile(*parts[:-1], type2="", ext=parts[-1])
    elif len(parts) == 6:
        csFile = CsFile(*parts)
    else:
        printError('Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts), fileName))

    return csFile


def getCoreImageFileName(filename):
    """Returns the part of Cityscapes filenames that is common to all data types

    e.g. for city_123456_123456_gtFine_polygons.json returns city_123456_123456
    """
    csFile = getCsFileInfo(filename)
    return "{}_{}_{}".format(csFile.city, csFile.sequenceNb, csFile.frameNb)


def getDirectory(fileName):
    """Returns the directory name for the given filename

    e.g.
    fileName = "/foo/bar/foobar.txt"
    return value is "bar"
    Not much error checking though
    """
    dirName = os.path.dirname(fileName)
    return os.path.basename(dirName)


def ensurePath(path):
    """Make sure that the given path exists"""
    if not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)


def writeDict2JSON(dictName, fileName):
    """Write a dictionary as json file"""
    with open(fileName, 'w') as f:
        f.write(json.dumps(dictName, default=lambda o: o.__dict__, sort_keys=True, indent=4))


# dummy main
if __name__ == "__main__":
    printError("Only for include, not executable on its own.")
