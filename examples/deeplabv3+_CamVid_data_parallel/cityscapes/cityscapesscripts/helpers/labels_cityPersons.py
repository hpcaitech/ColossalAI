#!/usr/bin/python
#
# CityPersons (cp) labels
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
LabelCp = namedtuple( 'LabelCp' , [

    'name'        , # The identifier of this label, e.g. 'pedestrian', 'rider', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# The 'ignore' label covers representations of humans, e.g. people on posters, reflections etc.
# Each annotation includes both the full bounding box (bbox) as well as a bounding box covering the visible area (bboxVis).
# The latter is obtained automatically from the segmentation masks.  

labelsCp = [
    #         name                     id   hasInstances   ignoreInEval   color
    LabelCp(  'ignore'               ,  0 , False        , True         , (250,170, 30) ),
    LabelCp(  'pedestrian'           ,  1 , True         , False        , (220, 20, 60) ),
    LabelCp(  'rider'                ,  2 , True         , False        , (  0,  0,142) ),
    LabelCp(  'sitting person'       ,  3 , True         , False        , (107,142, 35) ),
    LabelCp(  'person (other)'       ,  4 , True         , False        , (190,153,153) ),
    LabelCp(  'person group'         ,  5 , False        , True         , (255,  0,  0) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2labelCp      = { label.name    : label for label in labelsCp }
# id to label object
id2labelCp        = { label.id      : label for label in labelsCp }

