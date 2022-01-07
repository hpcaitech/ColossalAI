#!/usr/bin/python
#
# Classes to store, read, and write annotations
#

from __future__ import print_function, absolute_import, division
import os
import json
import numpy as np
from collections import namedtuple

# get current date and time
import datetime
import locale

from abc import ABCMeta, abstractmethod
from .box3dImageTransform import Camera

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])


class CsObjectType():
    """Type of an object"""
    POLY = 1  # polygon
    BBOX2D = 2  # bounding box
    BBOX3D = 3  # 3d bounding box
    IGNORE2D = 4  # 2d ignore region


class CsObject:
    """Abstract base class for annotation objects"""
    __metaclass__ = ABCMeta

    def __init__(self, objType):
        self.objectType = objType
        # the label
        self.label = ""

        # If deleted or not
        self.deleted = 0
        # If verified or not
        self.verified = 0
        # The date string
        self.date = ""
        # The username
        self.user = ""
        # Draw the object
        # Not read from or written to JSON
        # Set to False if deleted object
        # Might be set to False by the application for other reasons
        self.draw = True

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def fromJsonText(self, jsonText, objId=-1): pass

    @abstractmethod
    def toJsonText(self): pass

    def updateDate(self):
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.utf8')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'us_us.utf8')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'us_us')
        except Exception:
            pass
        self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Mark the object as deleted
    def delete(self):
        self.deleted = 1
        self.draw = False


class CsPoly(CsObject):
    """Class that contains the information of a single annotated object as polygon"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.POLY)
        # the polygon as list of points
        self.polygon = []
        # the object ID
        self.id = -1

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += '({},{}) '.format(p.x, p.y)
            else:
                polyText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[0].x, self.polygon[0].y,
                    self.polygon[1].x, self.polygon[1].y,
                    self.polygon[-2].x, self.polygon[-2].y,
                    self.polygon[-1].x, self.polygon[-1].y)
        else:
            polyText = "none"
        text = "Object: {} - {}".format(self.label, polyText)
        return text

    def fromJsonText(self, jsonText, objId=-1):
        self.id = objId
        self.label = str(jsonText['label'])
        self.polygon = [Point(p[0], p[1]) for p in jsonText['polygon']]
        if 'deleted' in jsonText.keys():
            self.deleted = jsonText['deleted']
        else:
            self.deleted = 0
        if 'verified' in jsonText.keys():
            self.verified = jsonText['verified']
        else:
            self.verified = 1
        if 'user' in jsonText.keys():
            self.user = jsonText['user']
        else:
            self.user = ''
        if 'date' in jsonText.keys():
            self.date = jsonText['date']
        else:
            self.date = ''
        if self.deleted == 1:
            self.draw = False
        else:
            self.draw = True

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['id'] = self.id
        objDict['deleted'] = self.deleted
        objDict['verified'] = self.verified
        objDict['user'] = self.user
        objDict['date'] = self.date
        objDict['polygon'] = []
        for pt in self.polygon:
            objDict['polygon'].append([pt.x, pt.y])

        return objDict


class CsBbox2d(CsObject):
    """Class that contains the information of a single annotated object as bounding box"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX2D)
        # the polygon as list of points
        self.bbox_amodal_xywh = []
        self.bbox_modal_xywh = []

        # the ID of the corresponding object
        self.instanceId = -1
        # the label of the corresponding object
        self.label = ""

    def __str__(self):
        bboxAmodalText = ""
        bboxAmodalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_amodal_xywh[0], self.bbox_amodal_xywh[1],  self.bbox_amodal_xywh[2],  self.bbox_amodal_xywh[3])

        bboxModalText = ""
        bboxModalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_modal_xywh[0], self.bbox_modal_xywh[1], self.bbox_modal_xywh[2], self.bbox_modal_xywh[3])

        text = "Object: {}\n - Amodal {}\n - Modal {}".format(
            self.label, bboxAmodalText, bboxModalText)
        return text

    def setAmodalBox(self, bbox_amodal):
        # sets the amodal box if required
        self.bbox_amodal_xywh = [
            bbox_amodal[0],
            bbox_amodal[1],
            bbox_amodal[2] - bbox_amodal[0],
            bbox_amodal[3] - bbox_amodal[1]
        ]

    # access 2d boxes in [xmin, ymin, xmax, ymax] format
    @property
    def bbox_amodal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_amodal_xywh[0],
            self.bbox_amodal_xywh[1],
            self.bbox_amodal_xywh[0] + self.bbox_amodal_xywh[2],
            self.bbox_amodal_xywh[1] + self.bbox_amodal_xywh[3]
        ]

    @property
    def bbox_modal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_modal_xywh[0],
            self.bbox_modal_xywh[1],
            self.bbox_modal_xywh[0] + self.bbox_modal_xywh[2],
            self.bbox_modal_xywh[1] + self.bbox_modal_xywh[3]
        ]

    def fromJsonText(self, jsonText, objId=-1):
        # try to load from cityperson format
        if 'bbox' in jsonText.keys() and 'bboxVis' in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['bbox']
            self.bbox_modal_xywh = jsonText['bboxVis']
        # both modal and amodal boxes are provided
        elif "modal" in jsonText.keys() and "amodal" in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['amodal']
            self.bbox_modal_xywh = jsonText['modal']
        # only amodal boxes are provided
        else:
            self.bbox_modal_xywh = jsonText['amodal']
            self.bbox_amodal_xywh = jsonText['amodal']

        # load label and instanceId if available
        if 'label' in jsonText.keys() and 'instanceId' in jsonText.keys():
            self.label = str(jsonText['label'])
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['modal'] = self.bbox_modal_xywh
        objDict['amodal'] = self.bbox_amodal_xywh

        return objDict


class CsBbox3d(CsObject):
    """Class that contains the information of a single annotated object as 3D bounding box"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX3D)

        self.bbox_2d = None

        self.center = []
        self.dims = []
        self.rotation = []
        self.instanceId = -1
        self.label = ""
        self.score = -1.

    def __str__(self):
        bbox2dText = str(self.bbox_2d)

        bbox3dText = ""
        bbox3dText += '\n - Center (x/y/z) [m]: {}/{}/{}'.format(
            self.center[0], self.center[1],  self.center[2])
        bbox3dText += '\n - Dimensions (l/w/h) [m]: {}/{}/{}'.format(
            self.dims[0], self.dims[1],  self.dims[2])
        bbox3dText += '\n - Rotation: {}/{}/{}/{}'.format(
            self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])

        text = "Object: {}\n2D {}\n - 3D {}".format(
            self.label, bbox2dText, bbox3dText)
        return text

    def fromJsonText(self, jsonText, objId=-1):
        # load 2D box
        self.bbox_2d = CsBbox2d()
        self.bbox_2d.fromJsonText(jsonText['2d'])

        self.center = jsonText['3d']['center']
        self.dims = jsonText['3d']['dimensions']
        self.rotation = jsonText['3d']['rotation']
        self.label = jsonText['label']
        self.score = jsonText['score']

        if 'instanceId' in jsonText.keys():
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['2d']['amodal'] = self.bbox_2d.bbox_amodal_xywh
        objDict['2d']['modal'] = self.bbox_2d.bbox_modal_xywh
        objDict['3d']['center'] = self.center
        objDict['3d']['dimensions'] = self.dims
        objDict['3d']['rotation'] = self.rotation

        return objDict

    @property
    def depth(self):
        # returns the BEV depth
        return np.sqrt(self.center[0]**2 + self.center[1]**2).astype(int)


class CsIgnore2d(CsObject):
    """Class that contains the information of a single annotated 2d ignore region"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.IGNORE2D)

        self.bbox_xywh = []
        self.label = ""
        self.instanceId = -1

    def __str__(self):
        bbox2dText = ""
        bbox2dText += 'Ignore Region:  (x1: {}, y1: {}), (w: {}, h: {})'.format(
            self.bbox_xywh[0], self.bbox_xywh[1], self.bbox_xywh[2], self.bbox_xywh[3])

        return bbox2dText

    def fromJsonText(self, jsonText, objId=-1):
        self.bbox_xywh = jsonText['2d']

        if 'label' in jsonText.keys():
            self.label = jsonText['label']

        if 'instanceId' in jsonText.keys():
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['2d'] = self.bbox_xywh

        return objDict

    @property
    def bbox(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_xywh[0],
            self.bbox_xywh[1],
            self.bbox_xywh[0] + self.bbox_xywh[2],
            self.bbox_xywh[1] + self.bbox_xywh[3]
        ]

    # Extend api to be compatible to bbox2d
    @property
    def bbox_amodal_xywh(self):
        return self.bbox_xywh

    @property
    def bbox_modal_xywh(self):
        return self.bbox_xywh


class Annotation:
    """The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox2d)"""

    # Constructor
    def __init__(self, objType=CsObjectType.POLY):
        # the width of that image and thus of the label image
        self.imgWidth = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []
        # the camera calibration
        self.camera = None
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects = []
        # load objects
        if self.objectType != CsObjectType.IGNORE2D:
            for objId, objIn in enumerate(jsonDict['objects']):
                if self.objectType == CsObjectType.POLY:
                    obj = CsPoly()
                elif self.objectType == CsObjectType.BBOX2D:
                    obj = CsBbox2d()
                elif self.objectType == CsObjectType.BBOX3D:
                    obj = CsBbox3d()
                obj.fromJsonText(objIn, objId)
                self.objects.append(obj)

        # load ignores
        if 'ignore' in jsonDict.keys():
            for ignoreId, ignoreIn in enumerate(jsonDict['ignore']):
                obj = CsIgnore2d()
                obj.fromJsonText(ignoreIn, ignoreId)
                self.objects.append(obj)

        # load camera calibration
        if 'sensor' in jsonDict.keys():
            self.camera = Camera(fx=jsonDict['sensor']['fx'],
                                 fy=jsonDict['sensor']['fy'],
                                 u0=jsonDict['sensor']['u0'],
                                 v0=jsonDict['sensor']['v0'],
                                 sensor_T_ISO_8855=jsonDict['sensor']['sensor_T_ISO_8855'])

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)

        return jsonDict

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())


# a dummy example
if __name__ == "__main__":
    obj = CsPoly()
    obj.label = 'car'
    obj.polygon.append(Point(0, 0))
    obj.polygon.append(Point(1, 0))
    obj.polygon.append(Point(1, 1))
    obj.polygon.append(Point(0, 1))

    print(type(obj).__name__)
    print(obj)
