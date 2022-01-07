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
# copy things
import copy
# numpy
import numpy as np
# matplotlib for colormaps
import matplotlib.colors
import matplotlib.cm
from PIL import Image

# import pyqt for everything graphical
from PyQt5 import QtCore, QtGui, QtWidgets

#################
# Helper classes
#################

from cityscapesscripts.helpers.version import version as VERSION

# annotation helpers
from cityscapesscripts.helpers.annotation import Annotation, CsObjectType
from cityscapesscripts.helpers.labels import name2label, assureSingleInstanceName
from cityscapesscripts.helpers.labels_cityPersons import name2labelCp
from cityscapesscripts.helpers.box3dImageTransform import Box3dImageTransform


from collections import namedtuple
LabelType = namedtuple('LabelType', 'description gtDir objectType')


class CsLabelType():
    """Viewing options for labels."""
    NONE = 0

    POLY_FINE = 1
    POLY_COARSE = 2

    CITYPERSONS_BBOX2D = 3

    CS3D_BBOX3D = 4
    CS3D_BBOX2D_MODAL = 5
    CS3D_BBOX2D_AMODAL = 6

    DISPARITY = 7

#################
# Main GUI class
#################


class CityscapesViewer(QtWidgets.QMainWindow):
    """The main class which is a QtGui -> Main Window"""

    #############################
    # Construction / Destruction
    #############################

    # Constructor
    def __init__(self):
        # Construct base class
        super(CityscapesViewer, self).__init__()

        # This is the configuration.

        # The filename of the image we currently working on
        self.currentFile = ""
        # The filename of the labels we currently working on
        self.currentLabelFile = ""
        # The path of the images of the currently loaded city
        self.city = ""
        # The name of the currently loaded city
        self.cityName = ""
        # The name of the current split
        self.split = ""
        # Ground truth type
        self.gtType = CsLabelType.NONE
        # The path of the labels. In this folder we expect a folder for each city
        # Within these city folders we expect the label with a filename matching
        # the images, except for the extension
        self.labelPath = ""
        # The transparency of the labels over the image
        self.transp = 0.5
        # The zoom toggle
        self.zoom = False
        # The zoom factor
        self.zoomFactor = 1.5
        # The size of the zoom window. Currently there is no setter or getter for that
        self.zoomSize = 400  # px

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
        self.gtExt = "_gt*.json"
        # Current image as QImage
        self.image = QtGui.QImage()
        # Index of the current image within the city folder
        self.idx = 0
        # All annotated objects in current image, i.e. list of csPoly or csBbox
        self.annotation = []
        # The current object the mouse points to. It's index in self.labels
        self.mouseObj = -1
        # The object that is highlighted and its label. An object instance
        self.highlightObj = None
        self.highlightObjLabel = None
        # The position of the mouse
        self.mousePosOrig = None
        # The position of the mouse scaled to label coordinates
        self.mousePosScaled = None
        # If the mouse is outside of the image
        self.mouseOutsideImage = True
        # The position of the mouse upon enabling the zoom window
        self.mousePosOnZoom = None
        # A list of toolbar actions that need an image
        self.actImage = []
        # A list of toolbar actions that need an image that is not the first
        self.actImageNotFirst = []
        # A list of toolbar actions that need an image that is not the last
        self.actImageNotLast = []
        # Toggle status of the play icon
        self.playState = False
        # Enable disparity visu in general
        self.enableDisparity = True
        # The filename of the disparity map we currently working on
        self.currentDispFile = ""
        # The disparity image
        self.dispImg = None
        # As overlay
        self.dispOverlay = None
        # The disparity search path
        self.dispPath = None
        # Disparity extension
        self.dispExt = "_disparity.png"
        # Available label types
        self.labelTypes = {
            CsLabelType.POLY_FINE: LabelType("gtFine", "gtFine", CsObjectType.POLY),
            CsLabelType.POLY_COARSE: LabelType("gtCoarse", "gtCoarse", CsObjectType.POLY),
            CsLabelType.CS3D_BBOX3D: LabelType("CS3D: 3D Boxes", "gtBbox3d", CsObjectType.BBOX3D),
            CsLabelType.CS3D_BBOX2D_MODAL: LabelType("CS3D: Modal 2D Boxes", "gtBbox3d", CsObjectType.BBOX3D),
            CsLabelType.CS3D_BBOX2D_AMODAL: LabelType("CS3D: Amodal 2D Boxes", "gtBbox3d", CsObjectType.BBOX3D),
            CsLabelType.CITYPERSONS_BBOX2D: LabelType("Citypersons", "gtBboxCityPersons", CsObjectType.BBOX2D),
            CsLabelType.DISPARITY: LabelType("Stereo Disparity", "disparity", CsObjectType.POLY)
        }

        # Generate colormap
        try:
            norm = matplotlib.colors.Normalize(vmin=3, vmax=100)
            cmap = matplotlib.cm.plasma
            self.colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        except Exception:
            self.enableDisparity = False

        # Default label
        self.defaultLabel = 'static'
        if self.defaultLabel not in name2label:
            print('The {0} label is missing in the internal label definitions.'.format(
                self.defaultLabel))
            return
        # Last selected label
        self.lastLabel = self.defaultLabel

        # Setup the GUI
        self.initUI()

        # If we already know a city from the saved config -> load it
        self.loadCity()
        self.imageChanged()

    # Destructor
    def __del__(self):
        return

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
        loadAction.triggered.connect(self.getCityFromUser)
        self.toolbar.addAction(loadAction)

        # Changing the label type
        labelTypeAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'label.png')), '&Tools', self)
        labelTypeAction.setShortcuts(['l'])
        self.setTip(labelTypeAction, 'Change label type')
        labelTypeAction.triggered.connect(self.getLabelTypeFromUser)
        self.toolbar.addAction(labelTypeAction)

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

        # Enable/disable zoom. Toggle button
        zoomAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'zoom.png')), '&Tools', self)
        zoomAction.setShortcuts(['z'])
        zoomAction.setCheckable(True)
        zoomAction.setChecked(self.zoom)
        self.setTip(zoomAction, 'Enable/disable permanent zoom')
        zoomAction.toggled.connect(self.zoomToggle)
        self.toolbar.addAction(zoomAction)
        self.actImage.append(zoomAction)

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

        # Display path to current image in message bar
        displayFilepathAction = QtWidgets.QAction(QtGui.QIcon(
            os.path.join(iconDir, 'filepath.png')), '&Tools', self)
        displayFilepathAction.setShortcut('f')
        self.setTip(displayFilepathAction, 'Show path to current image')
        displayFilepathAction.triggered.connect(self.displayFilepath)
        self.toolbar.addAction(displayFilepathAction)

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
        exitAction.setShortcuts(['Esc'])
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
        self.show()
        # Set a title
        self.applicationTitle = 'Cityscapes Viewer v{}'.format(VERSION)
        self.setWindowTitle(self.applicationTitle)
        self.displayHelpMessage()
        self.getCityFromUser()
        # And show the application
        self.show()

    #############################
    # Toolbar call-backs
    #############################

    # Switch to previous image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def prevImage(self):
        if not self.images:
            return
        if self.idx > 0:
            self.idx -= 1
            self.imageChanged()
        else:
            message = "Already at the first image"
            self.statusBar().showMessage(message)
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
            self.idx += 1
            self.imageChanged()
        elif self.playState:
            self.playState = False
            self.playAction.setChecked(False)
        else:
            message = "Already at the last image"
            self.statusBar().showMessage(message)
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)
        return

    # Play images, i.e. auto-switch to next image
    def playImages(self, status):
        self.playState = status
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)

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
        items = [os.path.basename(i) for i in self.images]
        (item, ok) = QtWidgets.QInputDialog.getItem(
            self, dlgTitle, "Image", items, self.idx, False)
        if (ok and item):
            idx = items.index(item)
            if idx != self.idx:
                self.idx = idx
                self.imageChanged()
        else:
            # Restore the message
            self.statusBar().showMessage(self.defaultStatusbar)

    # Toggle zoom

    def zoomToggle(self, status):
        self.zoom = status
        if status:
            self.mousePosOnZoom = self.mousePosOrig
        self.update()

    # Increase label transparency

    def minus(self):
        self.transp = max(self.transp-0.1, 0.0)
        self.update()

    def displayFilepath(self):
        self.statusBar().showMessage(
            "Current image: {0}".format(self.currentFile))
        self.update()

    def displayHelpMessage(self):

        message = self.applicationTitle + "\n\n"
        message += "INSTRUCTIONS\n"
        message += " - select a city and label type from drop-down menu\n"
        message += " - browse images and labels using\n"
        message += "   the toolbar buttons or the controls below\n"
        message += "\n"
        message += "CONTROLS\n"
        message += " - select city [o]\n"
        message += " - select label type [l]\n"
        message += " - highlight objects [move mouse]\n"
        message += " - next image [left arrow]\n"
        message += " - previous image [right arrow]\n"
        message += " - toggle autoplay [space]\n"
        message += " - increase/decrease label transparency\n"
        message += "   [ctrl+mousewheel] or [+ / -]\n"
        message += " - open zoom window [z]\n"
        message += "       zoom in/out [mousewheel]\n"
        message += "       enlarge/shrink zoom window [shift+mousewheel]\n"
        message += " - select a specific image [i]\n"
        message += " - show path to image below [f]\n"
        message += " - exit viewer [esc]\n"

        QtWidgets.QMessageBox.about(self, "HELP!", message)
        self.update()

    # Decrease label transparency

    def plus(self):
        self.transp = min(self.transp+0.1, 1.0)
        self.update()

    # Close the application
    def closeEvent(self, event):
        event.accept()

    #############################
    # Custom events
    #############################

    def imageChanged(self):
        # Load the first image
        self.loadImage()
        # Load its labels if available
        self.loadLabels()
        # Load disparities if available
        self.loadDisparities()
        # Update the object the mouse points to
        self.updateMouseObject()
        # Update the GUI
        self.update()

    #############################
    # File I/O
    #############################

    # Load the currently selected city if possible
    def loadCity(self):
        # clear annotations
        self.annotation = []
        # Search for all *.pngs to get the image list
        self.images = []
        if os.path.isdir(self.city):
            self.images = glob.glob(os.path.join(
                self.city, '*' + self.imageExt))
            self.images.sort()
            if self.currentFile in self.images:
                self.idx = self.images.index(self.currentFile)
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
            if not self.image.isNull() and filename == self.currentFile:
                success = True
            else:
                self.image = QtGui.QImage(filename)
                if self.image.isNull():
                    message = "Failed to read image: {0}".format(filename)
                else:
                    message = "Read image: {0}".format(filename)
                    self.currentFile = filename
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
        if not filename:
            self.clearAnnotation()
            return

        # If we have everything and the filename did not change, then we are good
        if self.annotation and filename == self.currentLabelFile:
            return

        # Clear the current labels first
        self.clearAnnotation()

        try:
            self.annotation = Annotation(self.labelTypes[self.gtType].objectType)
            self.annotation.fromJsonFile(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing labels in {0}. Message: {1}".format(
                filename, e.strerror)
            self.statusBar().showMessage(message)

        # Remember the filename loaded
        self.currentLabelFile = filename

        # Remember the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage(restoreMessage)

    # Load the disparity map from file
    # Only loads if they exist
    def loadDisparities(self):
        if not self.enableDisparity:
            return
        if not self.gtType == CsLabelType.DISPARITY:
            return

        filename = self.getDisparityFilename()
        if not filename:
            self.dispImg = None
            return

        # If we have everything and the filename did not change, then we are good
        if self.dispImg and filename == self.currentDispFile:
            return

        # Clear the current labels first
        self.dispImg = None

        try:
            self.dispImg = Image.open(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing disparities in {0}. Message: {1}".format(
                filename, e.strerror)
            self.statusBar().showMessage(message)
            self.dispImg = None

        if self.dispImg:
            dispNp = np.array(self.dispImg)
            dispNp = dispNp / 128.
            dispNp.round()
            dispNp = np.array(dispNp, dtype=np.uint8)

            dispQt = QtGui.QImage(
                dispNp.data, dispNp.shape[1], dispNp.shape[0], QtGui.QImage.Format_Indexed8)

            colortable = []
            for i in range(256):
                color = self.colormap.to_rgba(i)
                colorRgb = (int(color[0]*255),
                            int(color[1]*255), int(color[2]*255))
                colortable.append(QtGui.qRgb(*colorRgb))

            dispQt.setColorTable(colortable)
            dispQt = dispQt.convertToFormat(
                QtGui.QImage.Format_ARGB32_Premultiplied)
            self.dispOverlay = dispQt

        # Remember the filename loaded
        self.currentDispFile = filename

        # Remember the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage(restoreMessage)

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
        if self.gtType in [CsLabelType.CS3D_BBOX3D,
                           CsLabelType.CS3D_BBOX2D_MODAL,
                           CsLabelType.CS3D_BBOX2D_AMODAL]:
            overlay = self.draw3dLabels(qp)
        elif self.gtType in [CsLabelType.NONE,
                             CsLabelType.POLY_FINE,
                             CsLabelType.POLY_COARSE]:
            overlay = self.drawLabels(qp)
        elif self.gtType == CsLabelType.CITYPERSONS_BBOX2D:
            overlay = self.drawBboxes(qp)
        elif self.gtType == CsLabelType.DISPARITY:
            overlay = self.drawDisp(qp)
        # Draw the label name next to the mouse
        self.drawLabelAtMouse(qp)

        # Draw the zoom
        self.drawZoom(qp, overlay)

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
        # This variable we want to fill
        self.highlightObj = None

        # Without labels we cannot do so
        if not self.annotation:
            return

        # If available its the selected object
        highlightObjId = -1
        # If not available but the polygon is empty or closed, its the mouse object
        if highlightObjId < 0 and not self.mouseOutsideImage:
            highlightObjId = self.mouseObj
        # Get the actual object that is highlighted
        if highlightObjId >= 0:
            self.highlightObj = self.annotation.objects[highlightObjId]
            self.highlightObjLabel = self.annotation.objects[highlightObjId].label

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
            if obj.deleted:
                continue

            # The label of the object
            name = assureSingleInstanceName(obj.label)
            # If we do not know a color for this label, warn the user
            if name not in name2label:
                print("The annotations contain unknown labels. This should not happen. "
                      "Please inform the datasets authors. Thank you!")
                print("Details: label '{}', file '{}'".format(
                    name, self.currentLabelFile))
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
            if self.highlightObj and obj == self.highlightObj:
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
        if self.highlightObj:
            brush = QtGui.QBrush(QtCore.Qt.NoBrush)
            qp2.setBrush(brush)
            qp2.setPen(QtCore.Qt.DashLine)
            polyToDraw = self.getPolygon(
                self.highlightObj) * QtGui.QTransform.fromScale(self.scale, self.scale)
            qp2.drawPolygon(polyToDraw)

        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        qp.drawImage(self.xoff, self.yoff, overlay)
        # Restore settings
        qp.restore()

        return overlay

    def draw3dLabels(self, qp):
        if self.image.isNull() or self.w <= 0 or self.h <= 0:
            return
        if not self.annotation or self.gtType == CsLabelType.NONE:
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
        col = QtGui.QColor(0, 0, 0, 0)
        overlay.fill(col)

        # Create a new QPainter that draws in the overlay image
        qp2 = QtGui.QPainter()
        qp2.begin(overlay)

        # Loop through annotated objects
        for obj in self.annotation.objects:
            color = QtGui.QColor(*name2label[obj.label].color)

            # Draw only the 3D boxes
            if obj.objectType == CsObjectType.BBOX3D:
                if self.gtType == CsLabelType.CS3D_BBOX3D:
                    box3d_annotation = Box3dImageTransform(
                        camera=self.annotation.camera)
                    box3d_annotation.initialize_box_from_annotation(obj)
                    self.drawCityscapes3dBox3d(box3d_annotation, qp2, color, highlight=(obj == self.highlightObj))

                # Draw only the modal 2D boxes
                elif obj.bbox_2d.bbox_modal is not None and self.gtType == CsLabelType.CS3D_BBOX2D_MODAL:
                    bbox_modal = QtCore.QRectF(
                        obj.bbox_2d.bbox_modal_xywh[0],
                        obj.bbox_2d.bbox_modal_xywh[1],
                        obj.bbox_2d.bbox_modal_xywh[2],
                        obj.bbox_2d.bbox_modal_xywh[3])
                    self.drawCityscapes3dBox2d(bbox_modal, qp2, color, highlight=(obj == self.highlightObj))

                # Draw only the amodal 2D boxes
                elif obj.bbox_2d.bbox_amodal is not None and self.gtType == CsLabelType.CS3D_BBOX2D_AMODAL:
                    bbox = QtCore.QRectF(
                        obj.bbox_2d.bbox_amodal_xywh[0],
                        obj.bbox_2d.bbox_amodal_xywh[1],
                        obj.bbox_2d.bbox_amodal_xywh[2],
                        obj.bbox_2d.bbox_amodal_xywh[3])
                    self.drawCityscapes3dBox2d(bbox, qp2, color, highlight=(obj == self.highlightObj))

            # Draw only the ignore regions
            elif obj.objectType == CsObjectType.IGNORE2D:
                color = QtGui.QColor(*name2label[obj.label].color)
                bbox = QtCore.QRectF(
                    obj.bbox_xywh[0], obj.bbox_xywh[1], obj.bbox_xywh[2], obj.bbox_xywh[3])
                self.drawCityscapes3dBox2d(bbox, qp2, color, ignore=True, highlight=(obj == self.highlightObj))

        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        qp.drawImage(self.xoff, self.yoff, overlay)
        # Restore settings
        qp.restore()

        return overlay

    def drawCityscapes3dBox2d(self, bbox2d, qp, color, ignore=False, highlight=False):
        bboxToDraw = self.scaleBoundingBox(bbox2d)
        pen = QtGui.QPen(QtGui.QBrush(color), 3.0)
        qp.setPen(pen)
        if highlight:
            qp.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        else:
            if ignore:
                qp.setBrush(QtGui.QBrush(color, QtCore.Qt.DiagCrossPattern))
            else:
                color.setAlpha(60)
                qp.setBrush(QtGui.QBrush(color, QtCore.Qt.SolidPattern))
        qp.drawRect(bboxToDraw)

    def drawCityscapes3dBox3d(self, box3d_annotation, qp, color, highlight=False):
        box3d_sides = box3d_annotation.get_all_side_polygons_2d()
        box3d_side_visibilities = box3d_annotation.get_all_side_visibilities()
        box3d_sides.append(box3d_annotation.bottom_arrow_2d)
        box3d_side_visibilities.append(True)

        for i, (side, visible) in enumerate(zip(box3d_sides, box3d_side_visibilities)):
            side = [QtCore.QPointF(point[0], point[1]) for point in side]

            fill_color = color
            fill_color.setAlpha(60)

            polygon = QtGui.QPolygonF(
                side) * QtGui.QTransform.fromScale(self.scale, self.scale)

            if highlight:
                fill_brush = QtGui.QBrush(QtCore.Qt.NoBrush)
            else:
                fill_brush = QtGui.QBrush(fill_color, QtCore.Qt.SolidPattern)

            thickPen = QtGui.QPen(QtCore.Qt.SolidLine)
            pen_color = fill_color
            pen_color.setAlpha(255)
            thickPen.setColor(pen_color)
            if i < 6:
                thickPen.setWidth(2)
            else:
                # Make arrow thick
                thickPen.setWidth(5)

            if not visible:
                thickPen.setStyle(QtCore.Qt.CustomDashLine)
                thickPen.setDashPattern([1, 10])
                fill_brush = QtGui.QBrush(QtCore.Qt.NoBrush)

            qp.setPen(thickPen)
            qp.setBrush(fill_brush)
            qp.setRenderHints(QtGui.QPainter.Antialiasing)
            qp.drawPolygon(polygon)

    def getBoundingBox(self, obj):
        bboxAmodal = QtCore.QRectF(
            obj.bbox_amodal_xywh[0],
            obj.bbox_amodal_xywh[1],
            obj.bbox_amodal_xywh[2],
            obj.bbox_amodal_xywh[3])
        bboxModal = QtCore.QRectF(
            obj.bbox_modal_xywh[0],
            obj.bbox_modal_xywh[1],
            obj.bbox_modal_xywh[2],
            obj.bbox_modal_xywh[3])
        return bboxAmodal, bboxModal

    def scaleBoundingBox(self, bbox):
        bboxToDraw = copy.deepcopy(bbox)
        x, y, w, h = bboxToDraw.getRect()
        bboxToDraw.setTopLeft(QtCore.QPointF(x*self.scale, y*self.scale))
        bboxToDraw.setSize(QtCore.QSizeF(w*self.scale, h*self.scale))
        return bboxToDraw

    # Draw the labels in the given QPainter qp
    # optionally provide a list of labels to ignore
    def drawBboxes(self, qp, ignore=[]):
        if self.image.isNull() or self.w <= 0 or self.h <= 0:
            return
        if not self.annotation:
            return

        # The overlay is created in the viewing coordinates
        # This way, the drawing is more dense and the polygon edges are nicer
        # We create an image that is the overlay
        # Within this image we draw using another QPainter
        # Finally we use the real QPainter to overlay the overlay-image on what is drawn so far

        # The image that is used to draw the overlays
        overlay = QtGui.QImage(
            self.w, self.h, QtGui.QImage.Format_ARGB32_Premultiplied)
        # Fill the image
        col = QtGui.QColor(0, 0, 0, 0)
        overlay.fill(col)
        # Create a new QPainter that draws in the overlay image
        qp2 = QtGui.QPainter()
        qp2.begin(overlay)

        # Draw all objects
        for obj in self.annotation.objects:
            bboxAmodal, bboxModal = self.getBoundingBox(obj)
            bboxAmodalToDraw = self.scaleBoundingBox(bboxAmodal)
            bboxModalToDraw = self.scaleBoundingBox(bboxModal)
            # The label of the object
            name = obj.label
            # If we do not know a color for this label, warn the user
            if name not in name2labelCp:
                print("The annotations contain unknown labels. This should not happen. "
                      "Please inform the datasets authors. Thank you!")
                print("Details: label '{}', file '{}'".format(
                    name, self.currentLabelFile))
                continue

            # Reset brush for QPainter object
            qp2.setBrush(QtGui.QBrush())

            # Color from color table
            col = QtGui.QColor(*name2labelCp[name].color)

            if name2labelCp[name].hasInstances:
                if self.highlightObj and obj == self.highlightObj:
                    pen = QtGui.QPen(QtGui.QBrush(col), 5.0)
                else:
                    pen = QtGui.QPen(QtGui.QBrush(col), 3.0)
                qp2.setPen(pen)
                qp2.setOpacity(1.0)
                qp2.drawRect(bboxAmodalToDraw)

                if self.highlightObj and obj == self.highlightObj:
                    pen = QtGui.QPen(QtGui.QBrush(col), 3.0,
                                     style=QtCore.Qt.DotLine)
                    qp2.setPen(pen)
                    qp2.setOpacity(1.0)
                    qp2.drawRect(bboxModalToDraw)
                else:
                    pen = QtGui.QPen(QtGui.QBrush(col), 1.0,
                                     style=QtCore.Qt.DashLine)
                    qp2.setPen(pen)
                    qp2.setOpacity(1.0)
                    qp2.drawRect(bboxModalToDraw)

                    qp2.setBrush(QtGui.QBrush(col, QtCore.Qt.SolidPattern))
                    qp2.setOpacity(0.4)
                    qp2.drawRect(bboxModalToDraw)
            else:
                if self.highlightObj and obj == self.highlightObj:
                    pen = QtGui.QPen(QtGui.QBrush(col), 3.0)
                    qp2.setPen(pen)
                    qp2.setBrush(QtGui.QBrush(col, QtCore.Qt.NoBrush))
                else:
                    pen = QtGui.QPen(QtGui.QBrush(col), 3.0)
                    qp2.setPen(pen)
                    qp2.setBrush(QtGui.QBrush(col, QtCore.Qt.DiagCrossPattern))
                qp2.setOpacity(1.0)
                qp2.drawRect(bboxAmodalToDraw)

        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        qp.drawImage(self.xoff, self.yoff, overlay)
        # Restore settings
        qp.restore()

        return overlay

    # Draw the label name next to the mouse
    def drawLabelAtMouse(self, qp):
        # Nothing to do without a highlighted object
        if not self.highlightObj:
            return
        # Nothing to without a mouse position
        if not self.mousePosOrig:
            return

        # Save QPainter settings to stack
        qp.save()

        # That is the mouse position
        mouse = self.mousePosOrig

        # Will show zoom
        showZoom = self.zoom and not self.image.isNull() and self.w and self.h

        # The text that is written next to the mouse
        mouseText = self.highlightObj.label

        # Where to write the text
        # Depends on the zoom (additional offset to mouse to make space for zoom?)
        # The location in the image (if we are at the top we want to write below of the mouse)
        off = 36
        if showZoom:
            off += self.zoomSize/2
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
        rect.setTopLeft(QtCore.QPoint(mouse.x()-200, top))
        rect.setBottomRight(QtCore.QPoint(mouse.x()+200, btm))

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
        if not self.zoom:
            return
        # No image
        if self.image.isNull() or not self.w or not self.h:
            return
        # No mouse
        if not self.mousePosOrig:
            return

        # Abbreviation for the zoom window size
        zoomSize = self.zoomSize
        # Abbreviation for the mouse position
        mouse = self.mousePosOrig

        # The pixel that is the zoom center
        pix = self.mousePosScaled
        # The size of the part of the image that is drawn in the zoom window
        selSize = zoomSize / (self.zoomFactor * self.zoomFactor)
        # The selection window for the image
        sel = QtCore.QRectF(pix.x() - selSize/2, pix.y() -
                            selSize/2, selSize, selSize)
        # The selection window for the widget
        view = QtCore.QRectF(mouse.x()-zoomSize/2,
                             mouse.y()-zoomSize/2, zoomSize, zoomSize)
        if overlay:
            overlay_scaled = overlay.scaled(
                self.image.width(), self.image.height())
        else:
            overlay_scaled = QtGui.QImage(self.image.width(
            ), self.image.height(), QtGui.QImage.Format_ARGB32_Premultiplied)

        # Show the zoom image
        qp.save()
        qp.drawImage(view, self.image, sel)
        qp.setOpacity(self.transp)
        qp.drawImage(view, overlay_scaled, sel)
        qp.restore()

    # Draw disparities
    def drawDisp(self, qp):
        if not self.dispOverlay:
            return

        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        qp.drawImage(QtCore.QRect(self.xoff, self.yoff,
                                  self.w, self.h), self.dispOverlay)
        # Restore settings
        qp.restore()

        return self.dispOverlay
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

        mousePosOrig = QtCore.QPointF(event.x(), event.y())
        mousePosScaled = QtCore.QPointF(float(mousePosOrig.x(
        ) - self.xoff) / self.scale, float(mousePosOrig.y() - self.yoff) / self.scale)
        mouseOutsideImage = not self.image.rect().contains(mousePosScaled.toPoint())

        mousePosScaled.setX(max(mousePosScaled.x(), 0.))
        mousePosScaled.setY(max(mousePosScaled.y(), 0.))
        mousePosScaled.setX(min(mousePosScaled.x(), self.image.rect().right()))
        mousePosScaled.setY(
            min(mousePosScaled.y(), self.image.rect().bottom()))

        if not self.image.rect().contains(mousePosScaled.toPoint()):
            print(self.image.rect())
            print(mousePosScaled.toPoint())
            self.mousePosScaled = None
            self.mousePosOrig = None
            self.updateMouseObject()
            self.update()
            return

        self.mousePosScaled = mousePosScaled
        self.mousePosOrig = mousePosOrig
        self.mouseOutsideImage = mouseOutsideImage

        # Redraw
        self.updateMouseObject()
        self.update()

    # Mouse left the widget
    def leaveEvent(self, event):
        self.mousePosOrig = None
        self.mousePosScaled = None
        self.mouseOutsideImage = True

    # Mouse wheel scrolled
    def wheelEvent(self, event):
        ctrlPressed = event.modifiers() & QtCore.Qt.ControlModifier

        deltaDegree = event.angleDelta().y() / 8  # Rotation in degree
        deltaSteps = deltaDegree / 15  # Usually one step on the mouse is 15 degrees

        if ctrlPressed:
            self.transp = max(min(self.transp+(deltaSteps*0.1), 1.0), 0.0)
            self.update()
        else:
            if self.zoom:
                # If shift is pressed, change zoom window size
                if event.modifiers() and QtCore.Qt.Key_Shift:
                    self.zoomSize += deltaSteps * 10
                    self.zoomSize = max(self.zoomSize, 10)
                    self.zoomSize = min(self.zoomSize, 1000)
                # Change zoom factor
                else:
                    self.zoomFactor += deltaSteps * 0.05
                    self.zoomFactor = max(self.zoomFactor, 0.1)
                    self.zoomFactor = min(self.zoomFactor, 10)
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

    # Update the object that is selected by the current mouse curser
    def updateMouseObject(self):
        self.mouseObj = -1
        if self.mousePosScaled is None or self.annotation is None:
            return
        for idx in reversed(range(len(self.annotation.objects))):
            obj = self.annotation.objects[idx]
            if obj.objectType == CsObjectType.POLY:
                if self.getPolygon(obj).containsPoint(self.mousePosScaled, QtCore.Qt.OddEvenFill):
                    self.mouseObj = idx
                    break
            elif obj.objectType in [CsObjectType.BBOX2D, CsObjectType.IGNORE2D]:
                bbox, _ = self.getBoundingBox(obj)
                if bbox.contains(self.mousePosScaled):
                    self.mouseObj = idx
                    break
            elif obj.objectType == CsObjectType.BBOX3D:
                bbox, _ = self.getBoundingBox(obj.bbox_2d)
                if bbox.contains(self.mousePosScaled):
                    self.mouseObj = idx
                    break

    # Clear the current labels
    def clearAnnotation(self):
        self.annotation = None
        self.currentLabelFile = ""

    # Get the label type to view
    def getLabelTypeFromUser(self):
        if self.cityName == "" or self.split == "":
            return

        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        if 'CITYSCAPES_DATASET' in os.environ:
            csPath = os.environ['CITYSCAPES_DATASET']
        else:
            csPath = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), '..', '..')

        availableLabelTypes = []
        gtDirs = [os.path.basename(path) for path in glob.glob(os.path.join(csPath, '*'))]

        current_idx = 0
        for gtType in self.labelTypes:
            if self.labelTypes[gtType].gtDir in gtDirs:
                if gtType != CsLabelType.DISPARITY or self.enableDisparity:
                    availableLabelTypes.append(self.labelTypes[gtType].description)
                    # Preselect current label type
                    if self.gtType == gtType:
                        current_idx = len(availableLabelTypes) - 1

        # Specify title
        dlgTitle = "Select new label type"
        message = dlgTitle
        question = dlgTitle
        message = "Select label type for viewing"
        question = "Which label type would you like to view?"
        self.statusBar().showMessage(message)

        if availableLabelTypes:

            # Create and wait for dialog
            (item, ok) = QtWidgets.QInputDialog.getItem(self, dlgTitle, question,
                                                        availableLabelTypes, current_idx, False)

            # Restore message
            self.statusBar().showMessage(restoreMessage)

            if ok and item:
                self.gtType = [k for k, v in self.labelTypes.items() if v.description == item][0]

                if self.split == "test" and self.gtType != "disparity":
                    self.transp = 0.1
                else:
                    self.transp = 0.5

                self.city = os.path.normpath(os.path.join(
                    csPath, "leftImg8bit", self.split, self.cityName))
                self.labelPath = os.path.normpath(
                    os.path.join(csPath, self.labelTypes[self.gtType].gtDir, self.split, self.cityName))
                self.dispPath = os.path.normpath(
                    os.path.join(csPath, "disparity", self.split, self.cityName))

                self.loadCity()
                self.imageChanged()

    def getCityFromUser(self):
        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        if 'CITYSCAPES_DATASET' in os.environ:
            csPath = os.environ['CITYSCAPES_DATASET']
        else:
            csPath = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), '..', '..')

        availableCities = []
        splits = ["train_extra", "train", "val", "test"]
        for split in splits:
            cities = glob.glob(os.path.join(csPath, "leftImg8bit", split, '*'))
            cities.sort()
            availableCities.extend(
                [(split, os.path.basename(c)) for c in cities if os.listdir(c)])

        # List of possible labels
        items = [split + ", " + city for (split, city) in availableCities]

        # Preselect current city
        current_idx = 0
        cityNames = [item[1] for item in availableCities]
        if self.cityName in cityNames:
            current_idx = cityNames.index(self.cityName)

        # Specify title
        dlgTitle = "Select new city"
        message = dlgTitle
        question = dlgTitle
        message = "Select city for viewing"
        question = "Which city would you like to view?"
        self.statusBar().showMessage(message)

        if items:
            # Create and wait for dialog
            (item, ok) = QtWidgets.QInputDialog.getItem(self, dlgTitle, question,
                                                        items, current_idx, False)

            # Restore message
            self.statusBar().showMessage(restoreMessage)

            if ok and item:
                (split, city) = [str(i) for i in item.split(', ')]

                self.cityName = city
                self.split = split

                if self.gtType != CsLabelType.NONE:
                    self.city = os.path.normpath(os.path.join(
                        csPath, "leftImg8bit", self.split, self.cityName))
                    self.labelPath = os.path.normpath(
                        os.path.join(csPath, self.labelTypes[self.gtType].gtDir, self.split, self.cityName))
                    self.dispPath = os.path.normpath(
                        os.path.join(csPath, "disparity", self.split, self.cityName))
                    self.loadCity()
                    self.imageChanged()

                else:
                    self.getLabelTypeFromUser()

        else:

            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the Cityscapes root folder\n"
            warning += "or\n"
            warning += " - set CITYSCAPES_DATASET to the Cityscapes root folder\n"
            warning += "       e.g. 'export CITYSCAPES_DATASET=<root_path>'\n"

            reply = QtWidgets.QMessageBox.information(self, "ERROR!", warning,
                                                      QtWidgets.QMessageBox.Ok)
            if reply == QtWidgets.QMessageBox.Ok:
                sys.exit()

        return

    # Determine if the given candidate for a label path makes sense
    def isLabelPathValid(self, labelPath):
        return os.path.isdir(labelPath)

    # Get the filename where to load labels
    # Returns empty string if not possible
    def getLabelFilename(self):
        # And we need to have a directory where labels should be searched
        if not self.labelPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not self.isLabelPathValid(self.labelPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename(self.currentFile)
        filename = filename.replace(self.imageExt, self.gtExt)
        filename = os.path.join(self.labelPath, filename)
        search = glob.glob(filename)
        if not search:
            return ""
        filename = os.path.normpath(search[0])
        return filename

    # Get the filename where to load disparities
    # Returns empty string if not possible
    def getDisparityFilename(self):
        # And we need to have a directory where disparities should be searched
        if not self.dispPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not os.path.isdir(self.dispPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename(self.currentFile)
        filename = filename.replace(self.imageExt, self.dispExt)
        filename = os.path.join(self.dispPath, filename)
        filename = os.path.normpath(filename)
        return filename

    # Disable the popup menu on right click
    def createPopupMenu(self):
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    tool = CityscapesViewer()
    tool.resize(800, 510)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
