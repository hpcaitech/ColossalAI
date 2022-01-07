#!/usr/bin/env python
# -*- coding: utf-8 -*-


#################
# Import modules
#################

from __future__ import print_function, absolute_import, division
# get command line parameters
import sys
# walk directories
import glob
# access to OS functionality
import os
# (de)serialize config file
import json
# call processes
import subprocess
# get the user name
import getpass
# xml parsing
import xml.etree.ElementTree as ET
# copy stuff
import copy

# import pyqt for everything graphical
from PyQt5 import QtCore, QtGui, QtWidgets


#################
# Helper classes
#################

from cityscapesscripts.helpers.version import version as VERSION

# annotation helper
from cityscapesscripts.helpers.annotation import Point, Annotation, CsPoly
from cityscapesscripts.helpers.labels import name2label, assureSingleInstanceName

# Helper class that contains the current configuration of the Gui
# This config is loaded when started and saved when leaving


class configuration:
    # Constructor
    def __init__(self):
        # The filename of the image we currently working on
        self.currentFile = ""
        # The filename of the labels we currently working on
        self.currentLabelFile = ""
        # The filename of the corrections we currently working on
        self.currentCorrectionFile = ""
        # The path where the Cityscapes dataset is located
        self.csPath = ""
        # The path of the images of the currently loaded city
        self.city = ""
        # The name of the currently loaded city
        self.cityName = ""
        # The type of the current annotations
        self.gtType = ""
        # The split, where the currently loaded city belongs to
        self.split = ""
        # The path of the labels. In this folder we expect a folder for each city
        # Within these city folders we expect the label with a filename matching
        # the images, except for the extension
        self.labelPath = ""
        # The path to store correction markings
        self.correctionPath = ""
        # The transparency of the labels over the image
        self.transp = 0.5
        # The zoom toggle
        self.zoom = False
        # The zoom factor
        self.zoomFactor = 1.0
        # The size of the zoom window. Currently there is no setter or getter for that
        self.zoomSize = 400  # px
        # The highlight toggle
        self.highlight = False
        # The highlight label
        self.highlightLabelSelection = ""
        # Screenshot file
        self.screenshotFilename = "%i"
        # Correction mode
        self.correctionMode = False
        # Warn before saving that you are overwriting files
        self.showSaveWarning = True

    # Load from given filename
    def load(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                jsonText = f.read()
                jsonDict = json.loads(jsonText)
                for key in jsonDict:
                    if key in self.__dict__:
                        self.__dict__[key] = jsonDict[key]
        self.fixConsistency()

    # Make sure the config is consistent.
    # Automatically called after loading
    def fixConsistency(self):
        if self.currentFile:
            self.currentFile = os.path.normpath(self.currentFile)
        if self.currentLabelFile:
            self.currentLabelFile = os.path.normpath(self.currentLabelFile)
        if self.currentCorrectionFile:
            self.currentCorrectionFile = os.path.normpath(
                self.currentCorrectionFile)
        if self.csPath:
            self.csPath = os.path.normpath(self.csPath)
            if not os.path.isdir(self.csPath):
                self.csPath = ""
        if self.city:
            self.city = os.path.normpath(self.city)
            if not os.path.isdir(self.city):
                self.city = ""
        if self.labelPath:
            self.labelPath = os.path.normpath(self.labelPath)

        if self.correctionPath:
            self.correctionPath = os.path.normpath(self.correctionPath)

        if self.city:
            self.cityName == os.path.basename(self.city)

        if not os.path.isfile(self.currentFile) or os.path.dirname(self.currentFile) != self.city:
            self.currentFile = ""

        if not os.path.isfile(self.currentLabelFile) or \
           not os.path.isdir(os.path.join(self.labelPath, self.cityName)) or \
           os.path.dirname(self.currentLabelFile) != os.path.join(self.labelPath, self.cityName):
            self.currentLabelFile = ""

        if not os.path.isfile(self.currentCorrectionFile) or \
           not os.path.isdir(os.path.join(self.correctionPath, self.cityName)) or \
           os.path.dirname(self.currentCorrectionFile) != os.path.join(self.correctionPath, self.cityName):
            self.currentCorrectionFile = ""

    # Save to given filename (using pickle)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.__dict__,
                               default=lambda o: o.__dict__, sort_keys=True, indent=4))


def enum(**enums):
    return type('Enum', (), enums)


class CorrectionBox:

    types = enum(TO_CORRECT=1, TO_REVIEW=2, RESOLVED=3, QUESTION=4)

    def __init__(self, rect=None, annotation=""):
        self.type = CorrectionBox.types.TO_CORRECT
        self.bbox = rect
        self.annotation = annotation
        self.selected = False

        return

    def get_colour(self):
        if self.type == CorrectionBox.types.TO_CORRECT:
            return QtGui.QColor(255, 0, 0)
        elif self.type == CorrectionBox.types.TO_REVIEW:
            return QtGui.QColor(255, 255, 0)
        elif self.type == CorrectionBox.types.RESOLVED:
            return QtGui.QColor(0, 255, 0)
        elif self.type == CorrectionBox.types.QUESTION:
            return QtGui.QColor(0, 0, 255)

    def select(self):
        if not self.selected:
            self.selected = True
        return

    def unselect(self):
        if self.selected:
            self.selected = False
        return
    # Read the information from the given object node in an XML file
    # The node must have the tag object and contain all expected fields

    def readFromXMLNode(self, correctionNode):
        if not correctionNode.tag == 'correction':
            return

        typeNode = correctionNode.find('type')
        self.type = int(typeNode.text)
        annotationNode = correctionNode.find('annotation')
        self.annotation = annotationNode.text
        bboxNode = correctionNode.find('bbox')
        x = float(bboxNode.find('x').text)
        y = float(bboxNode.find('y').text)
        width = float(bboxNode.find('width').text)
        height = float(bboxNode.find('height').text)
        self.bbox = QtCore.QRectF(x, y, width, height)

    # Append the information to a node of an XML file
    # Creates an object node with all children and appends to the given node
    # Usually the given node is the root
    def appendToXMLNode(self, node):

        # New object node
        correctionNode = ET.SubElement(node, 'correction')
        correctionNode.tail = "\n"
        correctionNode.text = "\n"

        # Name node
        typeNode = ET.SubElement(correctionNode, 'type')
        typeNode.tail = "\n"
        typeNode.text = str(int(self.type))

        # Deleted node
        annotationNode = ET.SubElement(correctionNode, 'annotation')
        annotationNode.tail = "\n"
        annotationNode.text = str(self.annotation)

        # Polygon node
        bboxNode = ET.SubElement(correctionNode, 'bbox')
        bboxNode.text = "\n"
        bboxNode.tail = "\n"

        xNode = ET.SubElement(bboxNode, 'x')
        xNode.tail = "\n"
        yNode = ET.SubElement(bboxNode, 'y')
        yNode.tail = "\n"
        xNode.text = str(int(round(self.bbox.x())))
        yNode.text = str(int(round(self.bbox.y())))
        wNode = ET.SubElement(bboxNode, 'width')
        wNode.tail = "\n"
        hNode = ET.SubElement(bboxNode, 'height')
        hNode.tail = "\n"
        wNode.text = str(int(round(self.bbox.width())))
        hNode.text = str(int(round(self.bbox.height())))


#################
# Main GUI class
#################

# The main class which is a QtGui -> Main Window
class CityscapesLabelTool(QtWidgets.QMainWindow):

    #############################
    ## Construction / Destruction
    #############################

    # Constructor
    def __init__(self):
        # Construct base class
        super(CityscapesLabelTool, self).__init__()

        # The filename of where the config is saved and loaded
        configDir = os.path.dirname(__file__)
        self.configFile = os.path.join(configDir, "cityscapesLabelTool.conf")

        # This is the configuration.
        self.config = configuration()
        self.config.load(self.configFile)

        # Other member variables

        # The width that we actually use to show the image
        self.w = 0
        # The height that we actually use to show the image
        self.h = 0
        # The horizontal offset where we start drawing within the widget
        self.xoff = 0
        # The vertical offset where we start drawing withing the widget
        self.yoff = 0
        # A gap that we  leave around the image as little border
        self.bordergap = 20
        # The scale that was used, ie
        # self.w = self.scale * self.image.width()
        # self.h = self.scale * self.image.height()
        self.scale = 1.0
        # Filenames of all images in current city
        self.images = []
        # Image extension
        self.imageExt = "_leftImg8bit.png"
        # Ground truth extension
        self.gtExt = "{}_polygons.json"
        # Current image as QImage
        self.image = QtGui.QImage()
        # Index of the current image within the city folder
        self.idx = 0
        # All annotated objects in current image
        self.annotation = None
        # The XML ElementTree representing the corrections for the current image
        self.correctionXML = None
        # A list of changes that we did on the current annotation
        # Each change is simply a descriptive string
        self.changes = []
        # The current object the mouse points to. It's index in self.annotation.objects
        self.mouseObj = -1
        # The currently selected objects. Their index in self.annotation.objects
        self.selObjs = []
        # The objects that are highlighted. List of object instances
        self.highlightObjs = []
        # A label that is selected for highlighting
        self.highlightObjLabel = None
        # Texture for highlighting
        self.highlightTexture = None
        # The position of the mouse
        self.mousePos = None
        # TODO: NEEDS BETTER EXPLANATION/ORGANISATION
        self.mousePosOrig = None
        # The position of the mouse scaled to label coordinates
        self.mousePosScaled = None
        # If the mouse is outside of the image
        self.mouseOutsideImage = True
        # The position of the mouse upon enabling the zoom window
        self.mousePosOnZoom = None
        # The button state of the mouse
        self.mouseButtons = 0
        # A list of objects with changed layer
        self.changedLayer = []
        # A list of objects with changed polygon
        self.changedPolygon = []
        # A polygon that is drawn by the user
        self.drawPoly = QtGui.QPolygonF()
        # Treat the polygon as being closed
        self.drawPolyClosed = False
        # A point of this poly that is dragged
        self.draggedPt = -1
        # A list of toolbar actions that need an image
        self.actImage = []
        # A list of toolbar actions that need an image that is not the first
        self.actImageNotFirst = []
        # A list of toolbar actions that need an image that is not the last
        self.actImageNotLast = []
        # A list of toolbar actions that need changes
        self.actChanges = []
        # A list of toolbar actions that need a drawn polygon or selected objects
        self.actPolyOrSelObj = []
        # A list of toolbar actions that need a closed drawn polygon
        self.actClosedPoly = []
        # A list of toolbar actions that need selected objects
        self.actSelObj = []
        # A list of toolbar actions that need a single active selected object
        self.singleActSelObj = []
        # Toggle status of auto-doing screenshots
        self.screenshotToggleState = False
        # Toggle status of the play icon
        self.playState = False
        # Temporary zero transparency
        self.transpTempZero = False

        # Toggle correction mode on and off
        self.correctAction = []
        self.corrections = []
        self.selected_correction = -1

        self.in_progress_bbox = None
        self.in_progress_correction = None
        self.mousePressEvent = []

        # Default label
        self.defaultLabel = 'static'
        if not self.defaultLabel in name2label:
            print('The {0} label is missing in the internal label definitions.'.format(
                self.defaultLabel))
            return
        # Last selected label
        self.lastLabel = self.defaultLabel

        # Setup the GUI
        self.initUI()

        # Initially clear stuff
        self.deselectAllObjects()
        self.clearPolygon()
        self.clearChanges()

        # If we already know a city from the saved config -> load it
        self.loadCity()
        self.imageChanged()

    # Destructor
    def __del__(self):
        self.config.save(self.configFile)

    # Construct everything GUI related. Called by constructor
    def initUI(self):
        # Create a toolbar
        self.toolbar = self.addToolBar('Tools')

        # Add the tool buttons
        iconDir = os.path.join(os.path.dirname(__file__), 'icons')

        # Loading a new city
        loadAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'open.png')), '&Tools', self)
        loadAction.setShortcuts(['o'])
        self.setTip(loadAction, 'Open city')
        loadAction.triggered.connect(self.selectCity)
        self.toolbar.addAction(loadAction)

        # Open previous image
        backAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'back.png')), '&Tools', self)
        backAction.setShortcut('left')
        backAction.setStatusTip('Previous image')
        backAction.triggered.connect(self.prevImage)
        self.toolbar.addAction(backAction)
        self.actImageNotFirst.append(backAction)

        # Open next image
        nextAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'next.png')), '&Tools', self)
        nextAction.setShortcut('right')
        self.setTip(nextAction, 'Next image')
        nextAction.triggered.connect(self.nextImage)
        self.toolbar.addAction(nextAction)
        self.actImageNotLast.append(nextAction)

        # Play
        playAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'play.png')), '&Tools', self)
        playAction.setShortcut(' ')
        playAction.setCheckable(True)
        playAction.setChecked(False)
        self.setTip(playAction, 'Play all images')
        playAction.triggered.connect(self.playImages)
        self.toolbar.addAction(playAction)
        self.actImageNotLast.append(playAction)
        self.playAction = playAction

        # Select image
        selImageAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'shuffle.png')), '&Tools', self)
        selImageAction.setShortcut('i')
        self.setTip(selImageAction, 'Select image')
        selImageAction.triggered.connect(self.selectImage)
        self.toolbar.addAction(selImageAction)
        self.actImage.append(selImageAction)

        # Save the current image
        saveAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'save.png')), '&Tools', self)
        saveAction.setShortcut('s')
        self.setTip(saveAction, 'Save changes')
        saveAction.triggered.connect(self.save)
        self.toolbar.addAction(saveAction)
        self.actChanges.append(saveAction)

        # Clear the currently edited polygon
        clearPolAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'clearpolygon.png')), '&Tools', self)
        clearPolAction.setShortcuts(['q', 'Esc'])
        self.setTip(clearPolAction, 'Clear polygon')
        clearPolAction.triggered.connect(self.clearPolygonAction)
        self.toolbar.addAction(clearPolAction)
        self.actPolyOrSelObj.append(clearPolAction)

        # Create new object from drawn polygon
        newObjAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'newobject.png')), '&Tools', self)
        newObjAction.setShortcuts(['n'])
        self.setTip(newObjAction, 'New object')
        newObjAction.triggered.connect(self.newObject)
        self.toolbar.addAction(newObjAction)
        self.actClosedPoly.append(newObjAction)

        # Delete the currently selected object
        deleteObjectAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'deleteobject.png')), '&Tools', self)
        deleteObjectAction.setShortcuts(['d', 'delete'])
        self.setTip(deleteObjectAction, 'Delete object')
        deleteObjectAction.triggered.connect(self.deleteObject)
        self.toolbar.addAction(deleteObjectAction)
        self.actSelObj.append(deleteObjectAction)

        # Undo changes in current image, ie. reload labels from file
        undoAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'undo.png')), '&Tools', self)
        undoAction.setShortcut('u')
        self.setTip(undoAction, 'Undo all unsaved changes')
        undoAction.triggered.connect(self.undo)
        self.toolbar.addAction(undoAction)
        self.actChanges.append(undoAction)

        # Modify the label of a selected object
        labelAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'modify.png')), '&Tools', self)
        labelAction.setShortcuts(['m', 'l'])
        self.setTip(labelAction, 'Modify label')
        labelAction.triggered.connect(self.modifyLabel)
        self.toolbar.addAction(labelAction)
        self.actSelObj.append(labelAction)

        # Move selected object a layer up
        layerUpAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'layerup.png')), '&Tools', self)
        layerUpAction.setShortcuts(['Up'])
        self.setTip(layerUpAction, 'Move object a layer up')
        layerUpAction.triggered.connect(self.layerUp)
        self.toolbar.addAction(layerUpAction)
        self.singleActSelObj.append(layerUpAction)

        # Move selected object a layer down
        layerDownAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'layerdown.png')), '&Tools', self)
        layerDownAction.setShortcuts(['Down'])
        self.setTip(layerDownAction, 'Move object a layer down')
        layerDownAction.triggered.connect(self.layerDown)
        self.toolbar.addAction(layerDownAction)
        self.singleActSelObj.append(layerDownAction)

        # Enable/disable zoom. Toggle button
        zoomAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'zoom.png')), '&Tools', self)
        zoomAction.setShortcuts(['z'])
        zoomAction.setCheckable(True)
        zoomAction.setChecked(self.config.zoom)
        self.setTip(zoomAction, 'Enable/disable permanent zoom')
        zoomAction.toggled.connect(self.zoomToggle)
        self.toolbar.addAction(zoomAction)
        self.actImage.append(zoomAction)

        # Highlight objects of a certain class
        highlightAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'highlight.png')), '&Tools', self)
        highlightAction.setShortcuts(['g'])
        highlightAction.setCheckable(True)
        highlightAction.setChecked(self.config.highlight)
        self.setTip(highlightAction,
                    'Enable/disable highlight of certain object class')
        highlightAction.toggled.connect(self.highlightClassToggle)
        self.toolbar.addAction(highlightAction)
        self.actImage.append(highlightAction)

        # Decrease transparency
        minusAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'minus.png')), '&Tools', self)
        minusAction.setShortcut('-')
        self.setTip(minusAction, 'Decrease transparency')
        minusAction.triggered.connect(self.minus)
        self.toolbar.addAction(minusAction)

        # Increase transparency
        plusAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'plus.png')), '&Tools', self)
        plusAction.setShortcut('+')
        self.setTip(plusAction, 'Increase transparency')
        plusAction.triggered.connect(self.plus)
        self.toolbar.addAction(plusAction)

        # Take a screenshot
        screenshotAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'screenshot.png')), '&Tools', self)
        screenshotAction.setShortcut('t')
        self.setTip(screenshotAction, 'Take a screenshot')
        screenshotAction.triggered.connect(self.screenshot)
        self.toolbar.addAction(screenshotAction)
        self.actImage.append(screenshotAction)

        # Take a screenshot in each loaded frame
        screenshotToggleAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'screenshotToggle.png')), '&Tools', self)
        screenshotToggleAction.setShortcut('Ctrl+t')
        screenshotToggleAction.setCheckable(True)
        screenshotToggleAction.setChecked(False)
        self.setTip(screenshotToggleAction,
                    'Take a screenshot in each loaded frame')
        screenshotToggleAction.toggled.connect(self.screenshotToggle)
        self.toolbar.addAction(screenshotToggleAction)
        self.actImage.append(screenshotToggleAction)

        # Display path to current image in message bar
        displayFilepathAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'filepath.png')), '&Tools', self)
        displayFilepathAction.setShortcut('f')
        self.setTip(displayFilepathAction, 'Show path to current image')
        displayFilepathAction.triggered.connect(self.displayFilepath)
        self.toolbar.addAction(displayFilepathAction)

        # Open correction mode
        self.correctAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'checked6.png')), '&Tools', self)
        self.correctAction.setShortcut('c')
        self.correctAction.setCheckable(True)
        self.correctAction.setChecked(self.config.correctionMode)
        if self.config.correctionMode:
            self.correctAction.setIcon(QtGui.QIcon(
                os.path.join(iconDir, 'checked6_red.png')))
        self.setTip(self.correctAction, 'Toggle correction mode')
        self.correctAction.triggered.connect(self.toggleCorrectionMode)
        self.toolbar.addAction(self.correctAction)

        # Display help message
        helpAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'help19.png')), '&Tools', self)
        helpAction.setShortcut('h')
        self.setTip(helpAction, 'Help')
        helpAction.triggered.connect(self.displayHelpMessage)
        self.toolbar.addAction(helpAction)

        # Close the application
        exitAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'exit.png')), '&Tools', self)
        # exitAction.setShortcuts(['Esc'])
        self.setTip(exitAction, 'Exit')
        exitAction.triggered.connect(self.close)
        self.toolbar.addAction(exitAction)

        # The default text for the status bar
        self.defaultStatusbar = 'Ready'
        # Create a statusbar. Init with default
        self.statusBar().showMessage(self.defaultStatusbar)

        # Enable mouse move events
        self.setMouseTracking(True)
        self.toolbar.setMouseTracking(True)
        # Open in full screen
        screenShape = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(screenShape.width(), screenShape.height())
        # Set a title
        self.applicationTitle = 'Cityscapes Label Tool v{}'.format(VERSION)
        self.setWindowTitle(self.applicationTitle)
        # And show the application
        self.show()

    #############################
    # Toolbar call-backs
    #############################

    # The user pressed "select city"
    # The purpose of this method is to set these configuration attributes:
    #   - self.config.city           : path to the folder containing the images to annotate
    #   - self.config.cityName       : name of this folder, i.e. the city
    #   - self.config.labelPath      : path to the folder to store the polygons
    #   - self.config.correctionPath : path to store the correction boxes in
    #   - self.config.gtType         : type of ground truth, e.g. gtFine or gtCoarse
    #   - self.config.split          : type of split, e.g. train, val, test
    # The current implementation uses the environment variable 'CITYSCAPES_DATASET'
    # to determine the dataset root folder and search available data within.
    # Annotation types are required to start with 'gt', e.g. gtFine or gtCoarse.
    # To add your own annotations you could create a folder gtCustom with similar structure.
    #
    # However, this implementation could be easily changed to a completely different folder structure.
    # Just make sure to specify all three paths and a descriptive name as 'cityName'.
    # The gtType and split can be left empty.
    def selectCity(self):
        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        csPath = self.config.csPath
        if not csPath or not os.path.isdir(csPath):
            if 'CITYSCAPES_DATASET' in os.environ:
                csPath = os.environ['CITYSCAPES_DATASET']
            else:
                csPath = os.path.join(os.path.dirname(
                    os.path.realpath(__file__)), '..', '..')

        availableCities = []
        annotations = sorted(glob.glob(os.path.join(csPath, 'gt*')))
        annotations = [os.path.basename(a) for a in annotations]
        splits = ["train_extra", "train", "val", "test"]
        for gt in annotations:
            for split in splits:
                cities = glob.glob(os.path.join(csPath, gt, split, '*'))
                cities.sort()
                availableCities.extend(
                    [(split, gt, os.path.basename(c)) for c in cities if os.path.isdir(c)])

        # List of possible labels
        items = [split + ", " + gt + ", " +
                 city for (split, gt, city) in availableCities]
        # default
        previousItem = self.config.split + ", " + \
            self.config.gtType + ", " + self.config.cityName
        default = 0
        if previousItem in items:
            default = items.index(previousItem)

        # Specify title
        dlgTitle = "Select city"
        message = dlgTitle
        question = dlgTitle
        message = "Select city for editing"
        question = "Which city would you like to edit?"
        self.statusBar().showMessage(message)

        if items:

            # Create and wait for dialog
            (item, ok) = QtWidgets.QInputDialog.getItem(
                self, dlgTitle, question, items, default, False)

            # Restore message
            self.statusBar().showMessage(restoreMessage)

            if ok and item:
                (split, gt, city) = [str(i) for i in item.split(', ')]
                self.config.city = os.path.normpath(
                    os.path.join(csPath, "leftImg8bit", split, city))
                self.config.cityName = city

                self.config.labelPath = os.path.normpath(
                    os.path.join(csPath, gt, split, city))
                self.config.correctionPath = os.path.normpath(
                    os.path.join(csPath, gt+'_corrections', split, city))

                self.config.gtType = gt
                self.config.split = split

                self.deselectAllObjects()
                self.clearPolygon()
                self.loadCity()
                self.imageChanged()

        else:

            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the Cityscapes root folder\n"
            warning += "or\n"
            warning += " - set CITYSCAPES_DATASET to the Cityscapes root folder\n"
            warning += "       e.g. 'export CITYSCAPES_DATASET=<root_path>'\n"

            reply = QtWidgets.QMessageBox.information(
                self, "ERROR!", warning, QtWidgets.QMessageBox.Ok)
            if reply == QtWidgets.QMessageBox.Ok:
                sys.exit()

        return

    # Switch to previous image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def prevImage(self):
        if not self.images:
            return
        if self.idx > 0:
            if self.checkAndSave():
                self.idx -= 1
                self.imageChanged()
        return

    # Switch to next image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def nextImage(self):
        if not self.images:
            return
        if self.idx < len(self.images)-1:
            if self.checkAndSave():
                self.idx += 1
                self.imageChanged()
        elif self.playState:
            self.playState = False
            self.playAction.setChecked(False)

        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)
        return

    # Play images, i.e. auto-switch to next image
    def playImages(self, status):
        self.playState = status
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)

    # switch correction mode on and off

    def toggleCorrectionMode(self):
        if not self.config.correctionMode:
            self.config.correctionMode = True
            iconDir = os.path.join(os.path.dirname(sys.argv[0]), 'icons')
            self.correctAction.setIcon(QtGui.QIcon(
                os.path.join(iconDir, 'checked6_red.png')))
        else:
            self.config.correctionMode = False
            iconDir = os.path.join(os.path.dirname(sys.argv[0]), 'icons')
            self.correctAction.setIcon(QtGui.QIcon(
                os.path.join(iconDir, 'checked6.png')))
        self.update()
        return

    # Switch to a selected image of the file list
    # Ask the user for an image
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View

    def selectImage(self):
        if not self.images:
            return

        dlgTitle = "Select image to load"
        self.statusBar().showMessage(dlgTitle)
        items = ["{}: {}".format(num, os.path.basename(i))
                 for (num, i) in enumerate(self.images)]
        (item, ok) = QtWidgets.QInputDialog.getItem(
            self, dlgTitle, "Image", items, self.idx, False)
        if (ok and item):
            idx = items.index(item)
            if idx != self.idx and self.checkAndSave():
                self.idx = idx
                self.imageChanged()
        else:
            # Restore the message
            self.statusBar().showMessage(self.defaultStatusbar)

    # Save labels

    def save(self):
        # Status
        saved = False
        # Message to show at the status bar when done
        message = ""
        # Only save if there are changes, labels, an image filename and an image
        if self.changes and (self.annotation or self.corrections) and self.config.currentFile and self.image:
            if self.annotation:
                # set image dimensions
                self.annotation.imgWidth = self.image.width()
                self.annotation.imgHeight = self.image.height()

                # Determine the filename
                # If we have a loaded label file, then this is also the filename
                filename = self.config.currentLabelFile
                # If not, then generate one
                if not filename:
                    filename = self.getLabelFilename(True)

                if filename:
                    proceed = True
                    # warn user that he is overwriting an old file
                    if os.path.isfile(filename) and self.config.showSaveWarning:
                        msgBox = QtWidgets.QMessageBox(self)
                        msgBox.setWindowTitle("Overwriting")
                        msgBox.setText(
                            "Saving overwrites the original file and it cannot be reversed. Do you want to continue?")
                        msgBox.addButton(QtWidgets.QMessageBox.Cancel)
                        okAndNeverAgainButton = msgBox.addButton(
                            'OK and never ask again', QtWidgets.QMessageBox.AcceptRole)
                        okButton = msgBox.addButton(QtWidgets.QMessageBox.Ok)
                        msgBox.setDefaultButton(QtWidgets.QMessageBox.Ok)
                        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
                        msgBox.exec_()

                        # User clicked on "OK"
                        if msgBox.clickedButton() == okButton:
                            pass
                        # User clicked on "OK and never ask again"
                        elif msgBox.clickedButton() == okAndNeverAgainButton:
                            self.config.showSaveWarning = False
                        else:
                            # Do nothing
                            message += "Nothing saved, no harm has been done. "
                            proceed = False

                    # Save JSON file
                    if proceed:
                        try:
                            self.annotation.toJsonFile(filename)
                            saved = True
                            message += "Saved labels to {0} ".format(filename)
                        except IOError as e:
                            message += "Error writing labels to {0}. Message: {1} ".format(
                                filename, e.strerror)

                else:
                    message += "Error writing labels. Cannot generate a valid filename. "
            if self.corrections or self.config.currentCorrectionFile:
                # Determine the filename
                # If we have a loaded label file, then this is also the filename
                filename = self.config.currentCorrectionFile
                # If not, then generate one
                if not filename:
                    filename = self.getCorrectionFilename(True)

                if filename:
                    # Prepare the root
                    root = ET.Element('correction')
                    root.text = "\n"
                    root.tail = "\n"
                    # Add the filename of the image that is annotated
                    filenameNode = ET.SubElement(root, 'filename')
                    filenameNode.text = os.path.basename(
                        self.config.currentFile)
                    filenameNode.tail = "\n"
                    # Add the folder where this image is located in
                    # For compatibility with the LabelMe Tool, we need to use the folder
                    # StereoDataset/cityName
                    folderNode = ET.SubElement(root, 'folder')
                    folderNode.text = "StereoDataset/" + self.config.cityName
                    folderNode.tail = "\n"
                    # The name of the tool. Here, we do not follow the output of the LabelMe tool,
                    # since this is crap anyway
                    sourceNode = ET.SubElement(root, 'source')
                    sourceNode.text = "\n"
                    sourceNode.tail = "\n"
                    sourceImageNode = ET.SubElement(sourceNode, 'sourceImage')
                    sourceImageNode.text = "Label Cities"
                    sourceImageNode.tail = "\n"
                    sourceAnnotationNode = ET.SubElement(
                        sourceNode, 'sourceAnnotation')
                    sourceAnnotationNode.text = "mcLabelTool"
                    sourceAnnotationNode.tail = "\n"
                    # The image size
                    imagesizeNode = ET.SubElement(root, 'imagesize')
                    imagesizeNode.text = "\n"
                    imagesizeNode.tail = "\n"
                    nrowsNode = ET.SubElement(imagesizeNode, 'nrows')
                    nrowsNode.text = str(self.image.height())
                    nrowsNode.tail = "\n"
                    ncolsNode = ET.SubElement(imagesizeNode, 'ncols')
                    ncolsNode.text = str(self.image.height())
                    ncolsNode.tail = "\n"
                    # Add all objects
                    for correction in self.corrections:
                        correction.appendToXMLNode(root)

                    # Create the actual XML tree
                    self.correctionXML = ET.ElementTree(root)

                    # Save XML file
                    try:
                        self.correctionXML.write(filename)
                        saved = True
                        message += "Saved corrections to {0} ".format(filename)
                    except IOError as e:
                        message += "Error writing corrections to {0}. Message: {1} ".format(
                            filename, e.strerror)
                else:
                    message += "Error writing corrections. Cannot generate a valid filename. "
            # Clear changes
            if saved:
                self.clearChanges()
        else:
            message += "Nothing to save "
            saved = True

        # Show the status message
        self.statusBar().showMessage(message)

        return saved

    # Undo changes, ie. reload labels
    def undo(self):
        # check if we really want to do this in case there are multiple changes
        if len(self.changes) > 1:
            # Backup of status message
            restoreMessage = self.statusBar().currentMessage()
            # Create the dialog
            dlgTitle = "Undo changes?"
            self.statusBar().showMessage(dlgTitle)
            text = "Do you want to undo the following changes?\n"
            for c in self.changes:
                text += "- " + c + '\n'
            buttons = QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            ret = QtWidgets.QMessageBox.question(
                self, dlgTitle, text, buttons, QtWidgets.QMessageBox.Ok)
            proceed = False
            # If the user selected yes -> undo
            if ret == QtWidgets.QMessageBox.Ok:
                proceed = True
            self.statusBar().showMessage(restoreMessage)

            # If we do not proceed -> return
            if not proceed:
                return

        # Clear labels to force a reload
        self.annotation = None
        # Reload
        self.imageChanged()

    # Clear the drawn polygon and update
    def clearPolygonAction(self):
        self.deselectAllObjects()
        self.clearPolygon()
        self.update()

    # Create a new object from the current polygon
    def newObject(self):
        # Default label
        label = self.lastLabel

        # Ask the user for a label
        (label, ok) = self.getLabelFromUser(label)

        if ok and label:
            # Append and create the new object
            self.appendObject(label, self.drawPoly)

            # Clear the drawn polygon
            self.deselectAllObjects()
            self.clearPolygon()

            # Default message
            self.statusBar().showMessage(self.defaultStatusbar)

            # Set as default label for next time
            self.lastLabel = label

        # Redraw
        self.update()

    # Delete the currently selected object
    def deleteObject(self):
        # Cannot do anything without a selected object
        if not self.selObjs:
            return
        # Cannot do anything without labels
        if not self.annotation:
            return

        for selObj in self.selObjs:
            # The selected object that is deleted
            obj = self.annotation.objects[selObj]

        # Delete
        obj.delete()

        # Save changes
        self.addChange(
            "Deleted object {0} with label {1}".format(obj.id, obj.label))

        # Clear polygon
        self.deselectAllObjects()
        self.clearPolygon()

        # Redraw
        self.update()

    # Modify the label of a selected object
    def modifyLabel(self):
        # Cannot do anything without labels
        if not self.annotation:
            return
        # Cannot do anything without a selected object
        if not self.selObjs:
            return

        # The last selected object
        obj = self.annotation.objects[self.selObjs[-1]]
        # default label
        defaultLabel = obj.label
        defaultId = -1
        # If there is only one object the dialog text can be improved
        if len(self.selObjs) == 1:
            defaultId = obj.id

        (label, ok) = self.getLabelFromUser(defaultLabel, defaultId)

        if ok and label:
            for selObj in self.selObjs:
                # The selected object that is modified
                obj = self.annotation.objects[selObj]

                # Save changes
                if obj.label != label:
                    self.addChange("Set label {0} for object {1} with previous label {2}".format(
                        label, obj.id, obj.label))
                    obj.label = label
                    obj.updateDate()

        # Update
        self.update()

    # Move object a layer up
    def layerUp(self):
        # Change layer
        self.modifyLayer(+1)
        # Update
        self.update()

    # Move object a layer down
    def layerDown(self):
        # Change layer
        self.modifyLayer(-1)
        # Update
        self.update()

    # Toggle zoom
    def zoomToggle(self, status):
        self.config.zoom = status
        if status:
            self.mousePosOnZoom = self.mousePos
        self.update()

    # Toggle highlight
    def highlightClassToggle(self, status):
        if status:
            defaultLabel = ""
            if self.config.highlightLabelSelection and self.config.highlightLabelSelection in name2label:
                defaultLabel = self.config.highlightLabelSelection
            (label, ok) = self.getLabelFromUser(defaultLabel)

            if ok and label:
                self.config.highlightLabelSelection = label
            else:
                status = False

        self.config.highlight = status
        self.update()

    # Increase label transparency
    def minus(self):
        self.config.transp = max(self.config.transp-0.1, 0.0)
        self.update()

    def displayFilepath(self):
        self.statusBar().showMessage(
            "Current image: {0}".format(self.config.currentFile))
        self.update()

    # Decrease label transparency

    def plus(self):
        self.config.transp = min(self.config.transp+0.1, 1.0)
        self.update()

    # Take a screenshot
    def screenshot(self):
        # Get a filename for saving
        dlgTitle = "Get screenshot filename"
        filter = "Images (*.png *.xpm *.jpg)"
        answer, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, dlgTitle, self.config.screenshotFilename, filter, options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if answer:
            self.config.screenshotFilename = str(answer)
        else:
            return

        # Actually make the screenshot
        self.doScreenshot()

    # Toggle auto-making of screenshots
    def screenshotToggle(self, status):
        self.screenshotToggleState = status
        if status:
            self.screenshot()

    def displayHelpMessage(self):

        message = self.applicationTitle + "\n\n"
        message += "INSTRUCTIONS\n"
        message += " - press open (left button) to select a city from drop-down menu\n"
        message += " - browse images and edit labels using\n"
        message += "   the toolbar buttons (check tooltips) and the controls below\n"
        message += " - note that the editing happens in-place;\n"
        message += "   if you want to annotate your own images or edit a custom\n"
        message += "   set of labels, check (and modify) the code of the method 'loadCity'\n"
        message += " - note that this tool modifys the JSON polygon files, but\n"
        message += "   does not create or update the pngs; for the latter use\n"
        message += "   the preparation tools that come with this tool box.\n"
        message += "\n"
        message += "CONTROLS\n"
        message += " - highlight objects [move mouse]\n"
        message += " - draw new polygon\n"
        message += "     - start drawing a polygon [left click]\n"
        message += "     - add point to open polygon [left click]\n"
        message += "     - delete last added point [Backspace]\n"
        message += "     - close polygon [left click on first point]\n"
        message += " - select closed polygon, existing object [Ctrl + left click]\n"
        message += "     - move point [left click and hold on point, move mouse]\n"
        message += "     - add point [click on edge]\n"
        message += "     - delete point from polygon [Shift + left click on point]\n"
        message += "     - deselect polygon [Q]\n"
        message += "     - select multiple polygons [Ctrl + left click]\n"
        message += " - intersect/merge two polygons: draw new polygon, then\n"
        message += "     - intersect [Shift + left click on existing polygon]\n"
        message += "     - merge [Alt + left click on existing polygon]\n"
        message += " - open zoom window [Z or hold down right mouse button]\n"
        message += "     - zoom in/out [mousewheel]\n"
        message += "     - enlarge/shrink zoom window [shift+mousewheel]\n"
        message += " - start correction mode [C]\n"
        message += "     - draw a correction box [left click and hold, move, release]\n"
        message += "     - set box type [1,2,3,4]\n"
        message += "     - previous/next box [E,R]\n"
        message += "     - delete box [D]\n"
        message += "     - modify text, use ascii only [M]\n"

        QtWidgets.QMessageBox.about(self, "HELP!", message)
        self.update()

    # Close the application
    def closeEvent(self, event):
        if self.checkAndSave():
            event.accept()
        else:
            event.ignore()

    #############################
    # Custom events
    #############################

    def imageChanged(self):
        # Clear corrections
        self.corrections = []
        self.selected_correction = -1
        # Clear the polygon
        self.deselectAllObjects()
        self.clearPolygon()
        # Load the first image
        self.loadImage()
        # Load its labels if available
        self.loadLabels()
        # Load its corrections if available
        self.loadCorrections()
        # Update the object the mouse points to
        self.updateMouseObject()
        # Update the GUI
        self.update()
        # Save screenshot if set
        if self.screenshotToggleState:
            self.doScreenshot()

    #############################
    # File I/O
    #############################

    # Load the currently selected city if possible
    def loadCity(self):
        # Search for all *.pngs to get the image list
        self.images = []
        if os.path.isdir(self.config.city):
            self.images = glob.glob(os.path.join(
                self.config.city, '*' + self.imageExt))
            self.images.sort()
            if self.config.currentFile in self.images:
                self.idx = self.images.index(self.config.currentFile)
            else:
                self.idx = 0

    # Load the currently selected image
    # Does only load if not previously loaded
    # Does not refresh the GUI
    def loadImage(self):
        success = False
        message = self.defaultStatusbar
        if self.images:
            filename = self.images[self.idx]
            filename = os.path.normpath(filename)
            if not self.image.isNull() and filename == self.config.currentFile:
                success = True
            else:
                self.image = QtGui.QImage(filename)
                if self.image.isNull():
                    message = "Failed to read image: {0}".format(filename)
                else:
                    message = "Read image: {0}".format(filename)
                    self.config.currentFile = filename
                    success = True

        # Update toolbar actions that need an image
        for act in self.actImage:
            act.setEnabled(success)
        for act in self.actImageNotFirst:
            act.setEnabled(success and self.idx > 0)
        for act in self.actImageNotLast:
            act.setEnabled(success and self.idx < len(self.images)-1)

        self.statusBar().showMessage(message)

    # Load the labels from file
    # Only loads if they exist
    # Otherwise the filename is stored and that's it
    def loadLabels(self):
        filename = self.getLabelFilename()
        if not filename or not os.path.isfile(filename):
            self.clearAnnotation()
            return

        # If we have everything and the filename did not change, then we are good
        if self.annotation and filename == self.currentLabelFile:
            return

        # Clear the current labels first
        self.clearAnnotation()

        try:
            self.annotation = Annotation()
            self.annotation.fromJsonFile(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing labels in {0}. Message: {1}".format(
                filename, e.strerror)
            self.statusBar().showMessage(message)

        # Remember the filename loaded
        self.currentLabelFile = filename

        # Remeber the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage(restoreMessage)

    # Load the labels from file
    # Only loads if they exist
    # Otherwise the filename is stored and that's it
    def loadCorrections(self):  # TODO
        filename = self.getCorrectionFilename()
        if not filename:
            self.clearCorrections()
            return

        # If we have everything and the filename did not change, then we are good
        if self.correctionXML and self.corrections and filename == self.config.currentCorrectionFile:
            return

        # Clear the current labels first
        self.clearCorrections()

        # We do not always expect to have corrections, therefore prevent a failure due to missing file
        if not os.path.isfile(filename):
            return

        try:
            # Try to parse the XML file
            self.correctionXML = ET.parse(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing corrections in {0}. Message: {1}".format(
                filename, e.strerror)
            self.statusBar().showMessage(message)
            self.correctionXML = []
            return
        except ET.ParseError as e:
            # This is the error if the content is no valid XML
            message = "Error parsing corrections in {0}. Message: {1}".format(
                filename, e)
            self.statusBar().showMessage(message)
            self.correctionXML = []
            return

        # Remember the filename loaded
        self.config.currentCorrectionFile = filename

        # Remeber the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Iterate through all objects in the XML
        root = self.correctionXML.getroot()
        for i, objNode in enumerate(root.findall('correction')):
            # Instantate a new object and read the XML node
            obj = CorrectionBox()
            obj.readFromXMLNode(objNode)
            if i == 0:
                self.selected_correction = 0
                obj.select()

            # Append the object to our list of labels
            self.corrections.append(obj)

        # Restore the message
        self.statusBar().showMessage(restoreMessage)

    def modify_correction_type(self, correction_type):
        if self.selected_correction >= 0:
            self.corrections[self.selected_correction].type = correction_type
            self.addChange("Modified correction type.")
            self.update()
        return

    def delete_selected_annotation(self):
        if self.selected_correction >= 0 and self.config.correctionMode:
            del self.corrections[self.selected_correction]
            if self.selected_correction == len(self.corrections):
                self.selected_correction = self.selected_correction - 1
            if self.selected_correction >= 0:
                self.corrections[self.selected_correction].select()
            self.addChange("Deleted correction.")
            self.update()
        return

    def modify_correction_description(self):
        if self.selected_correction >= 0 and self.config.correctionMode:
            description = QtWidgets.QInputDialog.getText(self, "Modify Error Description", "Please describe the labeling error briefly.",
                                                         text=self.corrections[self.selected_correction].annotation)
            if description[1]:
                self.corrections[self.selected_correction].annotation = description[0]
                self.addChange("Changed correction description.")
                self.update()
        return

    def select_next_correction(self):

        if self.selected_correction >= 0:
            self.corrections[self.selected_correction].unselect()
            if self.selected_correction == (len(self.corrections) - 1):
                self.selected_correction = 0
            else:
                self.selected_correction = self.selected_correction + 1
            self.corrections[self.selected_correction].select()
            self.update()

        return

    def select_previous_correction(self):

        if self.selected_correction >= 0:
            self.corrections[self.selected_correction].unselect()
            if self.selected_correction == 0:
                self.selected_correction = (len(self.corrections) - 1)
            else:
                self.selected_correction = self.selected_correction - 1
            self.corrections[self.selected_correction].select()
            self.update()

        return

    #############################
    # Drawing
    #############################

    # This method is called when redrawing everything
    # Can be manually triggered by self.update()
    # Note that there must not be any other self.update within this method
    # or any methods that are called within

    def paintEvent(self, event):
        # Create a QPainter that can perform draw actions within a widget or image
        qp = QtGui.QPainter()
        # Begin drawing in the application widget
        qp.begin(self)
        # Update scale
        self.updateScale(qp)
        # Determine the object ID to highlight
        self.getHighlightedObject(qp)
        # Draw the image first
        self.drawImage(qp)
        # Draw the labels on top
        overlay = self.drawLabels(qp)
        # Draw the user drawn polygon
        self.drawDrawPoly(qp)
        self.drawDrawRect(qp)
        # Draw the label name next to the mouse
        self.drawLabelAtMouse(qp)
        # Draw the zoom
        # self.drawZoom(qp, overlay)
        self.drawZoom(qp, None)

        # Thats all drawing
        qp.end()

        # Forward the paint event
        QtWidgets.QMainWindow.paintEvent(self, event)

    # Update the scaling
    def updateScale(self, qp):
        if not self.image.width() or not self.image.height():
            return
        # Horizontal offset
        self.xoff = self.bordergap
        # Vertical offset
        self.yoff = self.toolbar.height()+self.bordergap
        # We want to make sure to keep the image aspect ratio and to make it fit within the widget
        # Without keeping the aspect ratio, each side of the image is scaled (multiplied) with
        sx = float(qp.device().width() - 2*self.xoff) / self.image.width()
        sy = float(qp.device().height() - 2*self.yoff) / self.image.height()
        # To keep the aspect ratio while making sure it fits, we use the minimum of both scales
        # Remember the scale for later
        self.scale = min(sx, sy)
        # These are then the actual dimensions used
        self.w = self.scale * self.image.width()
        self.h = self.scale * self.image.height()

    # Determine the highlighted object for drawing
    def getHighlightedObject(self, qp):
        # These variables we want to fill
        self.highlightObjs = []
        self.highlightObjLabel = None

        # Without labels we cannot do so
        if not self.annotation:
            return

        # If available set the selected objects
        highlightObjIds = self.selObjs
        # If not available but the polygon is empty or closed, its the mouse object
        if not highlightObjIds and (self.drawPoly.isEmpty() or self.drawPolyClosed) and self.mouseObj >= 0 and not self.mouseOutsideImage:
            highlightObjIds = [self.mouseObj]
        # Get the actual object that is highlighted
        if highlightObjIds:
            self.highlightObjs = [self.annotation.objects[i]
                                  for i in highlightObjIds]
        # Set the highlight object label if appropriate
        if self.config.highlight:
            self.highlightObjLabel = self.config.highlightLabelSelection
        elif len(highlightObjIds) == 1 and self.config.correctionMode:
            self.highlightObjLabel = self.annotation.objects[highlightObjIds[-1]].label

    # Draw the image in the given QPainter qp
    def drawImage(self, qp):
        # Return if no image available
        if self.image.isNull():
            return

        # Save the painters current setting to a stack
        qp.save()
        # Draw the image
        qp.drawImage(QtCore.QRect(self.xoff, self.yoff,
                                  self.w, self.h), self.image)
        # Restore the saved setting from the stack
        qp.restore()

    def getPolygon(self, obj):
        poly = QtGui.QPolygonF()
        for pt in obj.polygon:
            point = QtCore.QPointF(pt.x, pt.y)
            poly.append(point)
        return poly

    # Draw the labels in the given QPainter qp
    # optionally provide a list of labels to ignore
    def drawLabels(self, qp, ignore=[]):
        if self.image.isNull() or self.w <= 0 or self.h <= 0:
            return
        if not self.annotation:
            return
        if self.transpTempZero:
            return

        # The overlay is created in the viewing coordinates
        # This way, the drawing is more dense and the polygon edges are nicer
        # We create an image that is the overlay
        # Within this image we draw using another QPainter
        # Finally we use the real QPainter to overlay the overlay-image on what is drawn so far

        # The image that is used to draw the overlays
        overlay = QtGui.QImage(
            self.w, self.h, QtGui.QImage.Format_ARGB32_Premultiplied)
        # Fill the image with the default color
        defaultLabel = name2label[self.defaultLabel]
        col = QtGui.QColor(*defaultLabel.color)
        overlay.fill(col)
        # Create a new QPainter that draws in the overlay image
        qp2 = QtGui.QPainter()
        qp2.begin(overlay)

        # The color of the outlines
        qp2.setPen(QtGui.QColor('white'))
        # Draw all objects
        for obj in self.annotation.objects:
            # Some are flagged to not be drawn. Skip them
            if not obj.draw:
                continue

            # The label of the object
            name = assureSingleInstanceName(obj.label)
            # If we do not know a color for this label, warn the user
            if not name in name2label:
                print(
                    "The annotations contain unkown labels. This should not happen. Please inform the datasets authors. Thank you!")
                print("Details: label '{}', file '{}'".format(
                    name, self.currentLabelFile))
                continue

            # If we ignore this label, skip
            if name in ignore:
                continue

            poly = self.getPolygon(obj)

            # Scale the polygon properly
            polyToDraw = poly * \
                QtGui.QTransform.fromScale(self.scale, self.scale)

            # Default drawing
            # Color from color table, solid brush
            col = QtGui.QColor(*name2label[name].color)
            brush = QtGui.QBrush(col, QtCore.Qt.SolidPattern)
            qp2.setBrush(brush)
            # Overwrite drawing if this is the highlighted object
            if (obj in self.highlightObjs or name == self.highlightObjLabel):
                # First clear everything below of the polygon
                qp2.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
                qp2.drawPolygon(polyToDraw)
                qp2.setCompositionMode(
                    QtGui.QPainter.CompositionMode_SourceOver)
                # Set the drawing to a special pattern
                brush = QtGui.QBrush(col, QtCore.Qt.DiagCrossPattern)
                qp2.setBrush(brush)

            qp2.drawPolygon(polyToDraw)

        # Draw outline of selected object dotted
        for obj in self.highlightObjs:
            brush = QtGui.QBrush(QtCore.Qt.NoBrush)
            qp2.setBrush(brush)
            qp2.setPen(QtCore.Qt.DashLine)
            polyToDraw = self.getPolygon(
                obj) * QtGui.QTransform.fromScale(self.scale, self.scale)
            qp2.drawPolygon(polyToDraw)

        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.config.transp)
        # Draw the overlay image
        qp.drawImage(self.xoff, self.yoff, overlay)
        # Restore settings
        qp.restore()

        return overlay

    def drawDrawRect(self, qp):

        qp.save()
        qp.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        qp.setFont(QtGui.QFont('QFont::AnyStyle', 14))
        thickPen = QtGui.QPen()
        qp.setPen(thickPen)

        for c in self.corrections:
            rect = copy.deepcopy(c.bbox)

            width = rect.width()
            height = rect.height()
            rect.setX(c.bbox.x() * self.scale + self.xoff)
            rect.setY(c.bbox.y() * self.scale + self.yoff)

            rect.setWidth(width * self.scale)
            rect.setHeight(height * self.scale)

            if c.selected:
                thickPen.setColor(QtGui.QColor(0, 0, 0))
                if c.type == CorrectionBox.types.QUESTION:
                    descr = "QUESTION"
                elif c.type == CorrectionBox.types.RESOLVED:
                    descr = "FIXED"
                else:
                    descr = "ERROR"
                qp.setPen(thickPen)
                qp.drawText(QtCore.QPoint(self.xoff, self.yoff + self.h + 20),
                            "(%s: %s)" % (descr, c.annotation))
                pen_width = 6
            else:
                pen_width = 3

            colour = c.get_colour()
            thickPen.setColor(colour)
            thickPen.setWidth(pen_width)
            qp.setPen(thickPen)
            qp.drawRect(rect)

        if self.in_progress_bbox is not None:
            rect = copy.deepcopy(self.in_progress_bbox)
            width = rect.width()
            height = rect.height()
            rect.setX(self.in_progress_bbox.x() * self.scale + self.xoff)
            rect.setY(self.in_progress_bbox.y() * self.scale + self.yoff)

            rect.setWidth(width * self.scale)
            rect.setHeight(height * self.scale)

            thickPen.setColor(QtGui.QColor(255, 0, 0))
            thickPen.setWidth(3)
            qp.setPen(thickPen)
            qp.drawRect(rect)

        qp.restore()

    # Draw the polygon that is drawn and edited by the user
    # Usually the polygon must be rescaled properly. However when drawing
    # The polygon within the zoom, this is not needed. Therefore the option transform.
    def drawDrawPoly(self, qp, transform=None):
        # Nothing to do?
        if self.drawPoly.isEmpty():
            return
        if not self.image:
            return

        # Save QPainter settings to stack
        qp.save()

        # The polygon - make a copy
        poly = QtGui.QPolygonF(self.drawPoly)

        # Append the current mouse position
        if not self.drawPolyClosed and (self.mousePosScaled is not None):
            poly.append(self.mousePosScaled)

        # Transform
        if not transform:
            poly = poly * QtGui.QTransform.fromScale(self.scale, self.scale)
            poly.translate(self.xoff, self.yoff)
        else:
            poly = poly * transform

        # Do not fill the polygon
        qp.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))

        # Draw the polygon edges
        polyColor = QtGui.QColor(255, 0, 0)
        qp.setPen(polyColor)
        if not self.drawPolyClosed:
            qp.drawPolyline(poly)
        else:
            qp.drawPolygon(poly)

        # Get the ID of the closest point to the mouse
        if self.mousePosScaled is not None:
            closestPt = self.getClosestPoint(
                self.drawPoly, self.mousePosScaled)
        else:
            closestPt = (-1, -1)

        # If a polygon edge is selected, draw in bold
        if closestPt[0] != closestPt[1]:
            thickPen = QtGui.QPen(polyColor)
            thickPen.setWidth(3)
            qp.setPen(thickPen)
            qp.drawLine(poly[closestPt[0]], poly[closestPt[1]])

        # Draw the polygon points
        qp.setPen(polyColor)
        startDrawingPts = 0

        # A bit different if not closed
        if not self.drawPolyClosed:
            # Draw
            self.drawPoint(qp, poly.first(), True, closestPt ==
                           (0, 0) and self.drawPoly.size() > 1)
            # Do not draw again
            startDrawingPts = 1

        # The next in red
        for pt in range(startDrawingPts, poly.size()):
            self.drawPoint(
                qp, poly[pt], False, self.drawPolyClosed and closestPt == (pt, pt))

        # Restore QPainter settings from stack
        qp.restore()

    # Draw the label name next to the mouse
    def drawLabelAtMouse(self, qp):
        # Nothing to do without a highlighted object
        if not self.highlightObjs:
            return
        # Also we do not want to draw the label, if we have a drawn polygon
        if not self.drawPoly.isEmpty():
            return
        # Nothing to without a mouse position
        if not self.mousePos:
            return

        # Save QPainter settings to stack
        qp.save()

        # That is the mouse positiong
        mouse = self.mousePos

        # Will show zoom
        showZoom = self.config.zoom and not self.image.isNull() and self.w and self.h

        # The text that is written next to the mouse
        mouseText = self.highlightObjs[-1].label

        # Where to write the text
        # Depends on the zoom (additional offset to mouse to make space for zoom?)
        # The location in the image (if we are at the top we want to write below of the mouse)
        off = 36
        if showZoom:
            off += self.config.zoomSize/2
        if mouse.y()-off > self.toolbar.height():
            top = mouse.y()-off
            btm = mouse.y()
            vAlign = QtCore.Qt.AlignTop
        else:
            # The height of the cursor
            if not showZoom:
                off += 20
            top = mouse.y()
            btm = mouse.y()+off
            vAlign = QtCore.Qt.AlignBottom

        # Here we can draw
        rect = QtCore.QRect()
        rect.setTopLeft(QtCore.QPoint(mouse.x()-100, top))
        rect.setBottomRight(QtCore.QPoint(mouse.x()+100, btm))

        # The color
        qp.setPen(QtGui.QColor('white'))
        # The font to use
        font = QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold)
        qp.setFont(font)
        # Non-transparent
        qp.setOpacity(1)
        # Draw the text, horizontally centered
        qp.drawText(rect, QtCore.Qt.AlignHCenter | vAlign, mouseText)
        # Restore settings
        qp.restore()

    # Draw the zoom
    def drawZoom(self, qp, overlay):
        # Zoom disabled?
        if not self.config.zoom:
            return
        # No image
        if self.image.isNull() or not self.w or not self.h:
            return
        # No mouse
        if not self.mousePos:
            return

        # Abbrevation for the zoom window size
        zoomSize = self.config.zoomSize
        # Abbrevation for the mouse position
        mouse = self.mousePos

        # The pixel that is the zoom center
        pix = self.mousePosScaled
        # The size of the part of the image that is drawn in the zoom window
        selSize = zoomSize / (self.config.zoomFactor * self.config.zoomFactor)
        # The selection window for the image
        sel = QtCore.QRectF(pix.x() - selSize/2, pix.y() -
                            selSize/2, selSize, selSize)
        # The selection window for the widget
        view = QtCore.QRectF(mouse.x()-zoomSize/2,
                             mouse.y()-zoomSize/2, zoomSize, zoomSize)

        # Show the zoom image
        qp.drawImage(view, self.image, sel)

        # If we are currently drawing the polygon, we need to draw again in the zoom
        if not self.drawPoly.isEmpty():
            transform = QtGui.QTransform()
            quadFrom = QtGui.QPolygonF()
            quadFrom.append(sel.topLeft())
            quadFrom.append(sel.topRight())
            quadFrom.append(sel.bottomRight())
            quadFrom.append(sel.bottomLeft())
            quadTo = QtGui.QPolygonF()
            quadTo.append(view.topLeft())
            quadTo.append(view.topRight())
            quadTo.append(view.bottomRight())
            quadTo.append(view.bottomLeft())
            if QtGui.QTransform.quadToQuad(quadFrom, quadTo, transform):
                qp.setClipRect(view)
                # transform.translate(self.xoff,self.yoff)
                self.drawDrawPoly(qp, transform)
            else:
                print("not possible")

    #############################
    # Mouse/keyboard events
    #############################

    # Mouse moved
    # Need to save the mouse position
    # Need to drag a polygon point
    # Need to update the mouse selected object
    def mouseMoveEvent(self, event):
        if self.image.isNull() or self.w == 0 or self.h == 0:
            return

        self.updateMousePos(event.localPos())

        if not self.config.correctionMode:
            # If we are dragging a point, update
            if self.draggedPt >= 0:
                # Update the dragged point
                self.drawPoly.replace(self.draggedPt, self.mousePosScaled)
                # If the polygon is the polygon of the selected object,
                # update the object polygon and
                # keep track of the changes we do
                if self.selObjs:
                    obj = self.annotation.objects[self.selObjs[-1]]
                    obj.polygon[self.draggedPt] = Point(
                        self.mousePosScaled.x(), self.mousePosScaled.y())
                    # Check if we changed the object's polygon the first time
                    if not obj.id in self.changedPolygon:
                        self.changedPolygon.append(obj.id)
                        self.addChange(
                            "Changed polygon of object {0} with label {1}".format(obj.id, obj.label))
        else:
            if self.in_progress_bbox is not None:
                p0 = (self.mousePosScaled.x(), self.mousePosScaled.y())
                p1 = (self.mousePressEvent.x(), self.mousePressEvent.y())
                xy = min(p0[0], p1[0]), min(p0[1], p1[1])
                w, h = abs(p0[0] - p1[0]), abs(p0[1] - p1[1])
                self.in_progress_bbox = QtCore.QRectF(xy[0], xy[1], w, h)
            # p.set_x(xy[0])
            # p.set_y(xy[1])
            # p.set_width(w)
            # p.set_height(h)

        # Update the object selected by the mouse
        self.updateMouseObject()

        # Redraw
        self.update()

    # Mouse left the widget
    def leaveEvent(self, event):
        self.mousePos = None
        self.mousePosScaled = None
        self.mouseOutsideImage = True

    # Mouse button pressed
    # Start dragging of polygon point
    # Enable temporary toggling of zoom
    def mousePressEvent(self, event):

        self.mouseButtons = event.buttons()
        shiftPressed = QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier
        self.updateMousePos(event.localPos())
        self.mousePressEvent = self.mousePosScaled
        # Handle left click
        if event.button() == QtCore.Qt.LeftButton:

            # If the drawn polygon is closed and the mouse clicks a point,
            # Then this one is dragged around
            if not self.config.correctionMode:
                if self.drawPolyClosed and (self.mousePosScaled is not None):
                    closestPt = self.getClosestPoint(
                        self.drawPoly, self.mousePosScaled)
                    if shiftPressed:
                        if closestPt[0] == closestPt[1]:
                            del self.drawPoly[closestPt[0]]

                            # If the polygon is the polygon of the selected object,
                            # update the object
                            # and keep track of the changes we do
                            if self.selObjs:
                                obj = self.annotation.objects[self.selObjs[-1]]
                                del obj.polygon[closestPt[0]]
                                # Check if we changed the object's polygon the first time
                                if not obj.id in self.changedPolygon:
                                    self.changedPolygon.append(obj.id)
                                    self.addChange(
                                        "Changed polygon of object {0} with label {1}".format(obj.id, obj.label))

                            self.update()
                    else:
                        # If we got a point (or nothing), we make it dragged
                        if closestPt[0] == closestPt[1]:
                            self.draggedPt = closestPt[0]
                        # If we got an edge, we insert a point and make it dragged
                        else:
                            self.drawPoly.insert(
                                closestPt[1], self.mousePosScaled)
                            self.draggedPt = closestPt[1]
                            # If the polygon is the polygon of the selected object,
                            # update the object
                            # and keep track of the changes we do
                            if self.selObjs:
                                obj = self.annotation.objects[self.selObjs[-1]]
                                obj.polygon.insert(closestPt[1], Point(
                                    self.mousePosScaled.x(), self.mousePosScaled.y()))
                                # Check if we changed the object's polygon the first time
                                if not obj.id in self.changedPolygon:
                                    self.changedPolygon.append(obj.id)
                                    self.addChange(
                                        "Changed polygon of object {0} with label {1}".format(obj.id, obj.label))
            else:
                assert self.in_progress_bbox == None
                self.in_progress_bbox = QtCore.QRectF(
                    self.mousePosScaled.x(), self.mousePosScaled.y(), 0, 0)

        # Handle right click
        elif event.button() == QtCore.Qt.RightButton:
            self.toggleZoom(event.localPos())

        # Redraw
        self.update()

    # Mouse button released
    # End dragging of polygon
    # Select an object
    # Add a point to the polygon
    # Disable temporary toggling of zoom
    def mouseReleaseEvent(self, event):
        self.mouseButtons = event.buttons()
        ctrlPressed = event.modifiers() & QtCore.Qt.ControlModifier
        shiftPressed = event.modifiers() & QtCore.Qt.ShiftModifier
        altPressed = event.modifiers() & QtCore.Qt.AltModifier

        # Handle left click
        if event.button() == QtCore.Qt.LeftButton:
            if not self.config.correctionMode:
                # Check if Ctrl is pressed
                if ctrlPressed:
                    # If also Shift is pressed and we have a closed polygon, then we intersect
                    # the polygon with the mouse object
                    if shiftPressed and self.drawPolyClosed:
                        self.intersectPolygon()
                    # If also Alt is pressed and we have a closed polygon, then we merge
                    # the polygon with the mouse object
                    if altPressed and self.drawPolyClosed:
                        self.mergePolygon()
                    # Make the current mouse object the selected
                    # and process the selection
                    else:
                        self.selectObject()
                # Add the point to the drawn polygon if not already closed
                elif not self.drawPolyClosed:
                    # If the mouse would close the poly make sure to do so
                    if self.ptClosesPoly():
                        self.closePolygon()
                    elif self.mousePosScaled is not None:
                        if not self.drawPolyClosed and self.drawPoly.isEmpty():
                            self.mousePosOnZoom = self.mousePos
                        self.addPtToPoly(self.mousePosScaled)
                # Otherwise end a possible dragging
                elif self.drawPolyClosed:
                    self.draggedPt = -1
            else:
                if self.in_progress_bbox is not None:
                    if self.in_progress_bbox.width() > 20:
                        description = QtWidgets.QInputDialog.getText(
                            self, "Error Description", "Please describe the labeling error briefly.")
                        if description[1] and description[0]:
                            self.corrections.append(CorrectionBox(
                                self.in_progress_bbox, annotation=description[0]))
                            # last_annotation = self.in_progress_annotation  #TODO: self?
                            self.corrections[self.selected_correction].unselect(
                            )
                            self.selected_correction = len(self.corrections)-1
                            self.corrections[self.selected_correction].select()
                            self.addChange("Added correction.")
                    self.in_progress_annotation = None
                    self.in_progress_bbox = None

        # Handle right click
        elif event.button() == QtCore.Qt.RightButton:
            self.toggleZoom(event.localPos())

        # Redraw
        self.update()

    # Mouse wheel scrolled
    def wheelEvent(self, event):
        deltaDegree = event.angleDelta().y() / 8  # Rotation in degree
        deltaSteps = deltaDegree / 15  # Usually one step on the mouse is 15 degrees

        if self.config.zoom:
            # If shift is pressed, change zoom window size
            if event.modifiers() and QtCore.Qt.Key_Shift:
                self.config.zoomSize += deltaSteps * 10
                self.config.zoomSize = max(self.config.zoomSize, 10)
                self.config.zoomSize = min(self.config.zoomSize, 1000)
            # Change zoom factor
            else:
                self.config.zoomFactor += deltaSteps * 0.05
                self.config.zoomFactor = max(self.config.zoomFactor, 0.1)
                self.config.zoomFactor = min(self.config.zoomFactor, 10)
            self.update()

    # Key pressed
    def keyPressEvent(self, e):
        # Ctrl key changes mouse cursor
        if e.key() == QtCore.Qt.Key_Control:
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # Backspace deletes last point from polygon
        elif e.key() == QtCore.Qt.Key_Backspace:
            if not self.drawPolyClosed:
                del self.drawPoly[-1]
                self.update()
        # set alpha to temporary zero
        elif e.key() == QtCore.Qt.Key_0:
            self.transpTempZero = True
            self.update()
        elif e.key() == QtCore.Qt.Key_E:
            self.select_next_correction()
        elif e.key() == QtCore.Qt.Key_R:
            self.select_previous_correction()
        elif e.key() == QtCore.Qt.Key_1:
            self.modify_correction_type(CorrectionBox.types.TO_CORRECT)
        elif e.key() == QtCore.Qt.Key_2:
            self.modify_correction_type(CorrectionBox.types.TO_REVIEW)
        elif e.key() == QtCore.Qt.Key_3:
            self.modify_correction_type(CorrectionBox.types.RESOLVED)
        elif e.key() == QtCore.Qt.Key_4:
            self.modify_correction_type(CorrectionBox.types.QUESTION)
        elif e.key() == QtCore.Qt.Key_D and self.config.correctionMode:
            self.delete_selected_annotation()
        elif e.key() == QtCore.Qt.Key_M and self.config.correctionMode:
            self.modify_correction_description()

    # Key released
    def keyReleaseEvent(self, e):
        # Ctrl key changes mouse cursor
        if e.key() == QtCore.Qt.Key_Control:
            QtWidgets.QApplication.restoreOverrideCursor()
        # check for zero to release temporary zero
        # somehow, for the numpad key in some machines, a check on Insert is needed aswell
        elif e.key() == QtCore.Qt.Key_0 or e.key() == QtCore.Qt.Key_Insert:
            self.transpTempZero = False
            self.update()

    #############################
    # Little helper methods
    #############################

    # Helper method that sets tooltip and statustip
    # Provide an QAction and the tip text
    # This text is appended with a hotkeys and then assigned
    def setTip(self, action, tip):
        tip += " (Hotkeys: '" + \
            "', '".join([str(s.toString()) for s in action.shortcuts()]) + "')"
        action.setStatusTip(tip)
        action.setToolTip(tip)

    # Set the mouse positions
    # There are the original positions refering to the screen
    # Scaled refering to the image
    # And a zoom version, where the mouse movement is artificially slowed down
    def updateMousePos(self, mousePosOrig):
        if self.config.zoomFactor <= 1 or (self.drawPolyClosed or self.drawPoly.isEmpty()):
            sens = 1.0
        else:
            sens = 1.0/pow(self.config.zoomFactor, 3)

        if self.config.zoom and self.mousePosOnZoom is not None:
            mousePos = QtCore.QPointF(round((1-sens)*self.mousePosOnZoom.x() + (
                sens)*mousePosOrig.x()), round((1-sens)*self.mousePosOnZoom.y() + sens*mousePosOrig.y()))
        else:
            mousePos = mousePosOrig
        mousePosScaled = QtCore.QPointF(float(mousePos.x(
        ) - self.xoff) / self.scale, float(mousePos.y() - self.yoff) / self.scale)
        mouseOutsideImage = not self.image.rect().contains(mousePosScaled.toPoint())

        mousePosScaled.setX(max(mousePosScaled.x(), 0.))
        mousePosScaled.setY(max(mousePosScaled.y(), 0.))
        mousePosScaled.setX(min(mousePosScaled.x(), self.image.rect().right()))
        mousePosScaled.setY(
            min(mousePosScaled.y(), self.image.rect().bottom()))

        if not self.image.rect().contains(mousePosScaled.toPoint()):
            self.mousePos = None
            self.mousePosScaled = None
            self.mousePosOrig = None
            self.updateMouseObject()
            self.update()
            return

        self.mousePos = mousePos
        self.mousePosScaled = mousePosScaled
        self.mousePosOrig = mousePosOrig
        self.mouseOutsideImage = mouseOutsideImage

    # Toggle the zoom and update all mouse positions
    def toggleZoom(self, mousePosOrig):
        self.config.zoom = not self.config.zoom

        if self.config.zoom:
            self.mousePosOnZoom = self.mousePos
            # Update the mouse position afterwards
            self.updateMousePos(mousePosOrig)
        else:
            # Update the mouse position first
            self.updateMousePos(mousePosOrig)
            # Update the dragged point to the non-zoom point
            if not self.config.correctionMode and self.draggedPt >= 0:
                self.drawPoly.replace(self.draggedPt, self.mousePosScaled)

    # Get the point/edge index within the given polygon that is close to the given point
    # Returns (-1,-1) if none is close enough
    # Returns (i,i) if the point with index i is closed
    # Returns (i,i+1) if the edge from points i to i+1 is closest
    def getClosestPoint(self, poly, pt):
        closest = (-1, -1)
        distTh = 4.0
        dist = 1e9  # should be enough
        for i in range(poly.size()):
            curDist = self.ptDist(poly[i], pt)
            if curDist < dist:
                closest = (i, i)
                dist = curDist
        # Close enough?
        if dist <= distTh:
            return closest

        # Otherwise see if the polygon is closed, but a line is close enough
        if self.drawPolyClosed and poly.size() >= 2:
            for i in range(poly.size()):
                pt1 = poly[i]
                j = i+1
                if j == poly.size():
                    j = 0
                pt2 = poly[j]
                edge = QtCore.QLineF(pt1, pt2)
                normal = edge.normalVector()
                normalThroughMouse = QtCore.QLineF(
                    pt.x(), pt.y(), pt.x()+normal.dx(), pt.y()+normal.dy())
                intersectionPt = QtCore.QPointF()
                intersectionType = edge.intersect(
                    normalThroughMouse, intersectionPt)
                if intersectionType == QtCore.QLineF.BoundedIntersection:
                    curDist = self.ptDist(intersectionPt, pt)
                    if curDist < dist:
                        closest = (i, j)
                        dist = curDist

        # Close enough?
        if dist <= distTh:
            return closest

        # If we didnt return yet, we didnt find anything
        return (-1, -1)

    # Get distance between two points
    def ptDist(self, pt1, pt2):
        # A line between both
        line = QtCore.QLineF(pt1, pt2)
        # Length
        lineLength = line.length()
        return lineLength

    # Determine if the given point closes the drawn polygon (snapping)
    def ptClosesPoly(self):
        if self.drawPoly.isEmpty():
            return False
        if self.mousePosScaled is None:
            return False
        closestPt = self.getClosestPoint(self.drawPoly, self.mousePosScaled)
        return closestPt == (0, 0)

    # Draw a point using the given QPainter qp
    # If its the first point in a polygon its drawn in green
    # if not in red
    # Also the radius might be increased
    def drawPoint(self, qp, pt, isFirst, increaseRadius):
        # The first in green
        if isFirst:
            qp.setBrush(QtGui.QBrush(QtGui.QColor(
                0, 255, 0), QtCore.Qt.SolidPattern))
        # Other in red
        else:
            qp.setBrush(QtGui.QBrush(QtGui.QColor(
                255, 0, 0), QtCore.Qt.SolidPattern))

        # Standard radius
        r = 3.0
        # Increase maybe
        if increaseRadius:
            r *= 2.5
        # Draw
        qp.drawEllipse(pt, r, r)

    # Determine if the given candidate for a label path makes sense
    def isLabelPathValid(self, labelPath):
        return os.path.isdir(labelPath)

    # Ask the user to select a label
    # If you like, you can give an object ID for a better dialog texting
    # Note that giving an object ID assumes that its current label is the default label
    # If you dont, the message "Select new label" is used
    # Return is (label, ok). 'ok' is false if the user pressed Cancel
    def getLabelFromUser(self, defaultLabel="", objID=-1):
        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        # Update defaultLabel
        if not defaultLabel:
            defaultLabel = self.defaultLabel

        # List of possible labels
        items = list(name2label.keys())
        items.sort()
        default = items.index(defaultLabel)
        if default < 0:
            self.statusBar().showMessage(
                'The selected label is missing in the internal color map.')
            return

        # Specify title
        dlgTitle = "Select label"
        message = dlgTitle
        question = dlgTitle
        if objID >= 0:
            message = "Select new label for object {0} with current label {1}".format(
                objID, defaultLabel)
            question = "Label for object {0}".format(objID)
        self.statusBar().showMessage(message)

        # Create and wait for dialog
        (item, ok) = QtWidgets.QInputDialog.getItem(
            self, dlgTitle, question, items, default, False)

        # Process the answer a bit
        item = str(item)

        # Restore message
        self.statusBar().showMessage(restoreMessage)

        # Return
        return (item, ok)

    # Add a point to the drawn polygon
    def addPtToPoly(self, pt):
        self.drawPoly.append(pt)
        # Enable actions that need a polygon
        for act in self.actPolyOrSelObj:
            act.setEnabled(True)

    # Clear the drawn polygon
    def clearPolygon(self):
        # We do not clear, since the drawPoly might be a reference on an object one
        self.drawPoly = QtGui.QPolygonF()
        self.drawPolyClosed = False
        # Disable actions that need a polygon
        for act in self.actPolyOrSelObj:
            act.setEnabled(bool(self.selObjs))
        for act in self.actClosedPoly:
            act.setEnabled(False)

    # We just closed the polygon and need to deal with this situation
    def closePolygon(self):
        self.drawPolyClosed = True
        for act in self.actClosedPoly:
            act.setEnabled(True)
        message = "What should I do with the polygon? Press n to create a new object, "
        message += "press Ctrl + Shift + Left Click to intersect with another object, "
        message += "press Ctrl + Alt + Left Click to merge with another object."
        self.statusBar().showMessage(message)

    # Intersect the drawn polygon with the mouse object
    # and create a new object with same label and so on
    def intersectPolygon(self):
        # Cannot do anything without labels
        if not self.annotation:
            return
        # Cannot do anything without a single selected object
        if self.mouseObj < 0:
            return
        # The selected object that is modified
        obj = self.annotation.objects[self.mouseObj]

        # The intersection of the polygons
        intersection = self.drawPoly.intersected(self.getPolygon(obj))

        if not intersection.isEmpty():
            # Ask the user for a label
            self.drawPoly = intersection
            (label, ok) = self.getLabelFromUser(obj.label)

            if ok and label:
                # Append and create the new object
                self.appendObject(label, intersection)

                # Clear the drawn polygon
                self.clearPolygon()

                # Default message
                self.statusBar().showMessage(self.defaultStatusbar)

        # Deselect
        self.deselectAllObjects()
        # Redraw
        self.update()

    # Merge the drawn polygon with the mouse object
    # and create a new object with same label and so on
    def mergePolygon(self):
        # Cannot do anything without labels
        if not self.annotation:
            return
        # Cannot do anything without a single selected object
        if self.mouseObj < 0:
            return
        # The selected object that is modified
        obj = self.annotation.objects[self.mouseObj]

        # The union of the polygons
        union = self.drawPoly.united(self.getPolygon(obj))

        if not union.isEmpty():
            # Ask the user for a label
            self.drawPoly = union
            (label, ok) = self.getLabelFromUser(obj.label)

            if ok and label:
                # Append and create the new object
                self.appendObject(label, union)

                # Clear the drawn polygon
                self.clearPolygon()

                # Default message
                self.statusBar().showMessage(self.defaultStatusbar)

        # Deselect
        self.deselectAllObjects()
        # Redraw
        self.update()

    # Edit an object's polygon or clear the polygon if multiple objects are selected
    def initPolygonFromObject(self):
        # Cannot do anything without labels
        if not self.annotation:
            return
        # Cannot do anything without any selected object
        if not self.selObjs:
            return
        # If there are multiple objects selected, we clear the polygon
        if len(self.selObjs) > 1:
            self.clearPolygon()
            self.update()
            return

        # The selected object that is used for init
        obj = self.annotation.objects[self.selObjs[-1]]

        # Make a reference to the polygon
        self.drawPoly = self.getPolygon(obj)

        # Make sure its closed
        self.drawPolyClosed = True

        # Update toolbar icons
        # Enable actions that need a polygon
        for act in self.actPolyOrSelObj:
            act.setEnabled(True)
        # Enable actions that need a closed polygon
        for act in self.actClosedPoly:
            act.setEnabled(True)

        # Redraw
        self.update()

    # Create new object
    def appendObject(self, label, polygon):
        # Create empty annotation object
        # if first object
        if not self.annotation:
            self.annotation = Annotation()

        # Search the highest ID
        newID = 0
        for obj in self.annotation.objects:
            if obj.id >= newID:
                newID = obj.id + 1

        # New object
        # Insert the object in the labels list
        obj = CsPoly()
        obj.label = label

        obj.polygon = [Point(p.x(), p.y()) for p in polygon]

        obj.id = newID
        obj.deleted = 0
        obj.verified = 0
        obj.user = getpass.getuser()
        obj.updateDate()

        self.annotation.objects.append(obj)

        # Append to changes
        self.addChange(
            "Created object {0} with label {1}".format(newID, label))

        # Clear the drawn polygon
        self.deselectAllObjects()
        self.clearPolygon()

        # select the new object
        self.mouseObj = 0
        self.selectObject()

    # Helper for leaving an image
    # Returns true if the image can be left, false if not
    # Checks for possible changes and asks the user if they should be saved
    # If the user says yes, then they are saved and true is returned
    def checkAndSave(self):
        # Without changes it's ok to leave the image
        if not self.changes:
            return True

        # Backup of status message
        restoreMessage = self.statusBar().currentMessage()
        # Create the dialog
        dlgTitle = "Save changes?"
        self.statusBar().showMessage(dlgTitle)
        text = "Do you want to save the following changes?\n"
        for c in self.changes:
            text += "- " + c + '\n'
        buttons = QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
        ret = QtWidgets.QMessageBox.question(
            self, dlgTitle, text, buttons, QtWidgets.QMessageBox.Save)
        proceed = False
        # If the user selected yes -> save
        if ret == QtWidgets.QMessageBox.Save:
            proceed = self.save()
        # If the user selected to discard the changes, clear them
        elif ret == QtWidgets.QMessageBox.Discard:
            self.clearChanges()
            proceed = True
        # Otherwise prevent leaving the image
        else:
            proceed = False
        self.statusBar().showMessage(restoreMessage)
        return proceed

    # Actually save a screenshot
    def doScreenshot(self):
        # For creating the screenshot we re-use the label drawing function
        # However, we draw in an image using a QPainter

        # Create such an image
        img = QtGui.QImage(self.image)
        # Create a QPainter that can perform draw actions within a widget or image
        qp = QtGui.QPainter()
        # Begin drawing in the image
        qp.begin(img)

        # Remember some settings
        xoff = self.xoff
        yoff = self.yoff
        scale = self.scale
        w = self.w
        h = self.h
        # Update scale
        self.xoff = 0
        self.yoff = 0
        self.scale = 1
        self.w = self.image.width()
        self.h = self.image.height()
        # Detactivate the highlighted object
        self.highlightObjs = []

        # Blur the license plates
        # make this variabel a member and use as option if desired
        blurLicensePlates = True
        if blurLicensePlates:
            self.blurLicensePlates(qp)

        # Draw the labels on top
        ignore = []
        if blurLicensePlates:
            ignore.append('numberplate')
        self.drawLabels(qp, ignore)

        # Finish drawing
        qp.end()
        # Reset scale and stuff
        self.xoff = xoff
        self.yoff = yoff
        self.scale = scale
        self.w = w
        self.h = h

        # Generate the real filename for saving
        file = self.config.screenshotFilename
        # Replace occurance of %c with the city name (as directory)
        # Generate the directory if necessary
        cityIdx = file.find('%c')
        if cityIdx >= 0:
            if self.config.cityName:
                dir = os.path.join(file[:cityIdx], self.config.cityName)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                file = file.replace('%c', self.config.cityName + '/', 1)

                if file.find('%c') > 0:
                    message = "Found multiple '%c' in screenshot filename. Not allowed"
                    file = None
            else:
                message = "Do not have a city name. Cannot replace '%c' in screenshot filename."
                file = None
        # Replace occurances of %i with the image filename (without extension)
        if file:
            file = file.replace('%i', os.path.splitext(
                os.path.basename(self.config.currentFile))[0])
        # Add extension .png if no extension given
        if file:
            if not os.path.splitext(file)[1]:
                file += '.png'
        # Save
        if file:
            success = img.save(file)
            if success:
                message = "Saved screenshot to " + file
            else:
                message = "Failed to save screenshot"

        self.statusBar().showMessage(message)
        # Update to reset everything to the correct state
        self.update()

    # Blur the license plates
    # Argument is a qPainter
    # Thus, only use this method for screenshots.
    def blurLicensePlates(self, qp):
        # license plate name
        searchedNames = ['license plate']

        # the image
        img = self.image

        # Draw all objects
        for obj in self.annotation.objects:
            # Some are flagged to not be drawn. Skip them
            if not obj.draw:
                continue

            # The label of the object
            name = obj.label
            # If we do not know a color for this label, skip
            if name not in name2label:
                continue
            # If we do not blur this label, skip
            if not name in searchedNames:
                continue

            # Scale the polygon properly
            polyToDraw = self.getPolygon(
                obj) * QtGui.QTransform.fromScale(self.scale, self.scale)
            bb = polyToDraw.boundingRect()

            # Get the mean color within the polygon
            meanR = 0
            meanG = 0
            meanB = 0
            num = 0
            for y in range(max(int(bb.top()), 0), min(int(bb.bottom()+1.5), img.height())):
                for x in range(max(int(bb.left()), 0), min(int(bb.right()+1.5), img.width())):
                    col = img.pixel(x, y)
                    meanR += QtGui.QColor(col).red()
                    meanG += QtGui.QColor(col).green()
                    meanB += QtGui.QColor(col).blue()
                    num += 1
            meanR /= float(num)
            meanG /= float(num)
            meanB /= float(num)
            col = QtGui.QColor(meanR, meanG, meanB)
            qp.setPen(col)
            brush = QtGui.QBrush(col, QtCore.Qt.SolidPattern)
            qp.setBrush(brush)

            # Default drawing
            qp.drawPolygon(polyToDraw)

    # Update the object that is selected by the current mouse curser

    def updateMouseObject(self):
        self.mouseObj = -1
        if self.mousePosScaled is None:
            return
        if not self.annotation or not self.annotation.objects:
            return
        for idx in reversed(range(len(self.annotation.objects))):
            obj = self.annotation.objects[idx]
            if obj.draw and self.getPolygon(obj).containsPoint(self.mousePosScaled, QtCore.Qt.OddEvenFill):
                self.mouseObj = idx
                break

    # Print info about the currently selected object at the status bar
    def infoOnSelectedObject(self):
        if not self.selObjs:
            return
        objID = self.selObjs[-1]
        if self.annotation and objID >= 0:
            obj = self.annotation.objects[objID]
            self.statusBar().showMessage(
                "Label of object {0}: {1}".format(obj.id, obj.label))
        # else:
        #    self.statusBar().showMessage(self.defaultStatusbar)

    # Make the object selected by the mouse the real selected object
    def selectObject(self):
        # If there is no mouse selection, we are good
        if self.mouseObj < 0:
            self.deselectObject()
            return

        # Append the object to selection if it's not in there
        if not self.mouseObj in self.selObjs:
            self.selObjs.append(self.mouseObj)
        # Otherwise remove the object
        else:
            self.deselectObject()

        # update polygon
        self.initPolygonFromObject()

        # If we have selected objects make the toolbar actions active
        if self.selObjs:
            for act in self.actSelObj + self.actPolyOrSelObj:
                act.setEnabled(True)
        # If we have a single selected object make their toolbar actions active
        for act in self.singleActSelObj:
            act.setEnabled(len(self.selObjs) == 1)

        self.infoOnSelectedObject()

    # Deselect object
    def deselectObject(self):
        # If there is no object to deselect, we are good
        if not self.selObjs:
            return
        # If the mouse does not select and object, remove the last one
        if self.mouseObj < 0:
            del self.selObjs[-1]
        # Otherwise try to find the mouse obj in the list
        if self.mouseObj in self.selObjs:
            self.selObjs.remove(self.mouseObj)

        # No object left?
        if not self.selObjs:
            for act in self.actSelObj:
                act.setEnabled(False)
            for act in self.actPolyOrSelObj:
                act.setEnabled(bool(self.drawPoly))
        # If we have a single selected object make their toolbar actions active
        for act in self.singleActSelObj:
            act.setEnabled(len(self.selObjs) == 1)
        self.infoOnSelectedObject()

    # Deselect all objects
    def deselectAllObjects(self):
        # If there is no object to deselect, we are good
        self.selObjs = []
        self.mouseObj = -1
        for act in self.actSelObj:
            act.setEnabled(False)
        # If we have a single selected object make their toolbar actions active
        for act in self.singleActSelObj:
            act.setEnabled(len(self.selObjs) == 1)
        self.infoOnSelectedObject()

    # Modify the layer of the selected object
    # Move the layer up (negative offset) or down (postive offset)
    def modifyLayer(self, offset):
        # Cannot do anything without labels
        if not self.annotation:
            return
        # Cannot do anything without a single selected object
        if len(self.selObjs) != 1:
            return

        # The selected object that is modified
        obj = self.annotation.objects[self.selObjs[-1]]
        # The index in the label list we are right now
        oldidx = self.selObjs[-1]
        # The index we want to move to
        newidx = oldidx + offset

        # Make sure not not exceed zero and the list
        newidx = max(newidx, 0)
        newidx = min(newidx, len(self.annotation.objects)-1)

        # If new and old idx are equal, there is nothing to do
        if oldidx == newidx:
            return

        # Move the entry in the labels list
        self.annotation.objects.insert(
            newidx, self.annotation.objects.pop(oldidx))

        # Update the selected object to the new index
        self.selObjs[-1] = newidx
        self.statusBar().showMessage(
            "Moved object {0} with label {1} to layer {2}".format(obj.id, obj.label, newidx))

        # Check if we moved the object the first time
        if not obj.id in self.changedLayer:
            self.changedLayer.append(obj.id)
            self.addChange(
                "Changed layer for object {0} with label {1}".format(obj.id, obj.label))

    # Add a new change
    def addChange(self, text):
        if not text:
            return

        self.changes.append(text)
        for act in self.actChanges:
            act.setEnabled(True)

    # Clear list of changes
    def clearChanges(self):
        self.changes = []
        self.changedLayer = []
        self.changedPolygon = []
        for act in self.actChanges:
            act.setEnabled(False)

    # Clear the current labels
    def clearAnnotation(self):
        self.annotation = None
        self.clearChanges()
        self.deselectAllObjects()
        self.clearPolygon()
        self.config.currentLabelFile = ""

    def clearCorrections(self):
        self.correctionXML = None
        self.corrections = []
        # self.clearChanges() #TODO perhaps?
        # self.clearPolygon()
        self.config.currentCorrectionFile = ""

    # Get the filename where to load/save labels
    # Returns empty string if not possible
    # Set the createDirs to true, if you want to create needed directories
    def getLabelFilename(self, createDirs=False):
        # We need the name of the current city
        if not self.config.cityName:
            return ""
        # And we need to have a directory where labels should be searched
        if not self.config.labelPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.config.currentFile:
            return ""
        # Check if the label directory is valid. This folder is selected by the user
        # and thus expected to exist
        if not self.isLabelPathValid(self.config.labelPath):
            return ""
        # Dirs are not automatically created in this version of the tool
        if not os.path.isdir(self.config.labelPath):
            return ""

        labelDir = self.config.labelPath

        # extension of ground truth files
        if self.config.gtType:
            ext = self.gtExt.format('_'+self.config.gtType)
        else:
            ext = self.gtExt.format('')
        # Generate the filename of the label file
        filename = os.path.basename(self.config.currentFile)
        filename = filename.replace(self.imageExt, ext)
        filename = os.path.join(labelDir, filename)
        filename = os.path.normpath(filename)
        return filename

    # Get the filename where to load/save labels
    # Returns empty string if not possible
    # Set the createDirs to true, if you want to create needed directories
    def getCorrectionFilename(self, createDirs=False):
        # And we need to have a directory where corrections are stored
        if not self.config.correctionPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.config.currentFile:
            return ""

        # Folder where to store the labels
        correctionDir = self.config.correctionPath

        # If the folder does not exist, create it if allowed
        if not os.path.isdir(correctionDir):
            if createDirs:
                os.makedirs(correctionDir)
                if not os.path.isdir(correctionDir):
                    return ""
            else:
                return ""

        # Generate the filename of the label file
        filename = os.path.basename(self.config.currentFile)
        filename = filename.replace(self.imageExt, '.xml')
        filename = os.path.join(correctionDir, filename)
        filename = os.path.normpath(filename)
        return filename

    # Disable the popup menu on right click
    def createPopupMenu(self):
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    tool = CityscapesLabelTool()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
