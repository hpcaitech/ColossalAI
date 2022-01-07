#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to label images,
# where each pixel has an ID that represents the ground truth label.
#
# Usage: json2labelImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdLabelImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt

# Image processing
from PIL import Image
from PIL import ImageDraw

# cityscapes imports
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import name2label

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print('Reads labels as polygons in JSON format and converts them to label images,')
    print('where each pixel has an ID that represents the ground truth label.')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image
def createLabelImage(annotation, encoding, outline=None):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    # the background
    if encoding == "ids":
        background = name2label['unlabeled'].id
    elif encoding == "trainIds":
        background = name2label['unlabeled'].trainId
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # If the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # If the ID is negative that polygon should not be drawn
        if name2label[label].id < 0:
            continue

        if encoding == "ids":
            val = name2label[label].id
        elif encoding == "trainIds":
            val = name2label[label].trainId
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:
                drawer.polygon( polygon, fill=val, outline=outline )
            else:
                drawer.polygon( polygon, fill=val )
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    return labelImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the label image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
#     - "color"    : classes are encoded using the corresponding colors
def json2labelImg(inJson,outImg,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    labelImg   = createLabelImage( annotation , encoding )
    labelImg.save( outImg )

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2labelImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if len(args) == 0:
        printError( "Missing input json file" )
    elif len(args) == 1:
        printError( "Missing output image filename" )
    elif len(args) > 2:
        printError( "Too many arguments" )

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2labelImg( inJson , outImg , "trainIds" )
    else:
        json2labelImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
