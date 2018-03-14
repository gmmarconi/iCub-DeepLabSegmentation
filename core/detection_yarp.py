#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import tensorflow as tf
from matplotlib import gridspec
from istantiate_model import DeepLabModel
import get_dataset_colormap
import os, sys


import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image


yarp_path = '/home/gmarconi/coding/yarp/yarp/build/lib/python'
if yarp_path not in sys.path:
    sys.path.insert(0, yarp_path)
    
import yarp

# Initialise YARP
yarp.Network.init()

class Detector (yarp.RFModule):
    def __init__(self, in_port_name, out_det_img_port_name, out_det_port_name, rpc_thresh_port_name, out_img_port_name, classes, image_w, image_h, deeplabmodel, prototxt, cpu_mode, gpu_id):

        if tf.__version__ < '1.5.0':
            raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

        self.LABEL_NAMES = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tv'
        ])

        self.FULL_LABEL_MAP = np.arange(len(self.LABEL_NAMES)).reshape(len(self.LABEL_NAMES), 1)
        self.FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(self.FULL_LABEL_MAP)

        self.deeplabmodel = deeplabmodel
        #self._TARBALL_NAME = 'deeplab_model.tar.gz'
        print(self.LABEL_NAMES)

        self.model = DeepLabModel(self.deeplabmodel)


        # Images port initialization
        ## Prepare ports

        self._in_port = yarp.BufferedPortImageRgb()
        #  self._in_port = yarp.Port()
        self._in_port_name = in_port_name
        self._in_port.open(self._in_port_name)

        self._out_det_port = yarp.BufferedPortBottle()
        self._out_det_port_name = out_det_port_name
        self._out_det_port.open(self._out_det_port_name)

        self._out_det_img_port = yarp.Port()
        self._out_det_img_port_name = out_det_img_port_name
        self._out_det_img_port.open(self._out_det_img_port_name)

        self._out_img_port = yarp.Port()
        self._out_img_port_name = out_img_port_name
        self._out_img_port.open(self._out_img_port_name)

        ## Prepare image buffers
        ### Input
        print 'prepare input image'

        #self._in_buf_array = np.ones((image_h, image_w, 3), dtype = np.uint8)
        self._in_buf_array = Image.new(mode='RGB', size=(image_w, image_h))

        self._in_buf_image = yarp.ImageRgb()
        self._in_buf_image.resize(image_w, image_h)
        self._in_buf_image.setExternal(self._in_buf_array, self._in_buf_array.shape[1], self._in_buf_array.shape[0])

        ### Output
        print 'prepare output image'
        self._out_buf_image = yarp.ImageRgb()
        self._out_buf_image.resize(image_w, image_h)
        #self._out_buf_array = np.zeros((image_h, image_w, 3), dtype = np.uint8)
        self._out_buf_array = Image.new(mode='RGB', size=(image_w, image_h))

        self._out_buf_image.setExternal(self._out_buf_array, self._out_buf_array.shape[1], self._out_buf_array.shape[0])


def vis_segmentation(self, image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(Detector.FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), Detector.LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)

def updateModule(self):
    cmd = yarp.Bottle()
    reply = yarp.Bottle()
    print 'reading cmd in updateModule\n'
    self._rpc_thresh_port.read(cmd, willReply=True)
    if cmd.size() is 1:
        raw_input('press any key to continue')
        print 'cmd size 1\n'
        self._set_threshold(cmd, reply)
        self._rpc_thresh_port.reply(reply)
    else:
        raw_input('press any key to continue')
        print 'cmd size != 1\n'
        ans = 'Received bottle has invalid size of ' + cmd.size()
        reply.addString(ans)
        self._rpc_thresh_port.reply(reply)

def _sendDetections(self, frame, dets):
    print 'sending detections...'

    detection = self._out_det_port.prepare()
    # frame_to_send = self._out_img_port.prepare()

    detection.clear()
    # frame_to_send.clear()

    # frame_to_send = frame
    stamp = yarp.Stamp()
    for i in range(len(dets)):
        for j in range(len(dets[i])):
            d = dets[i][j]
            cls_id = int(d[5])

            det_list = detection.addList()

            det_list.addDouble(d[0])
            det_list.addDouble(d[1])
            det_list.addDouble(d[2])
            det_list.addDouble(d[3])
            det_list.addDouble(d[4])
            det_list.addString(self._classes[cls_id])

    self._out_det_port.setEnvelope(stamp)
    self._out_img_port.setEnvelope(stamp)

    # self._out_det_port.write(detection)
    # self._out_img_port.write(frame_to_send)
    self._out_det_port.write()
    self._out_img_port.write(frame)


def cleanup(self):
    print 'cleanup'
    self._in_port.close()
    self._out_det_img_port.close()
    self._out_det_port.close()
    self._rpc_thresh_port.close()
    self._out_img_port.close()

def run(self, cpu_mode, vis=False):

    while(True):
        # Read an image from the port
        # self._in_port.read(self._in_buf_image)
        received_image = self._in_port.read()
        # self._in_buf_image = received_image
        # received_image.copy(self._in_buf_image)
        self._in_buf_image.copy(received_image)

        assert self._in_buf_array.__array_interface__['data'][0] == self._in_buf_image.getRawImage().__long__()

        frame = self._in_buf_array

        out_img, segmentation_map = self.model.run(frame)

        vis_segmentation(out_img, segmentation_map)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='DeepLab demo')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')

    #arguments for internal ports
    parser.add_argument('--inport', dest='in_port_name', help='input port',
                        default='/pyfaster:in')
    parser.add_argument('--outdetimgport', dest='out_det_img_port_name', help='output port for detected images',
                        default='/pyfaster:detimgout')
    parser.add_argument('--outdetsport', dest='out_det_port_name', help='output port for detections',
                        default='/pyfaster:detout')
    parser.add_argument('--outimgport', dest='out_img_port_name', help='output port for images',
                        default='/pyfaster:imgout')
    #arguments for external ports
    parser.add_argument('--viewerport', dest='viewer_port_name', help='port to send detected image',
                        default='/pyfaster:vis')
    parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
                        default='/depthCamera/rgbImage:o')
    # parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
    #                     choices=NETS.keys(), default='/yarprealsense/coulour:o')
    parser.add_argument('--thresh_port', dest='rpc_thresh_port_name', help='rpc port name where to set detection threshold',
                        default='/pyfaster:thresh')

    parser.add_argument('--deeplab_model', dest='deeplab_model', help='path to the deeplab model',
                        default='')
    parser.add_argument('--classes_file', dest='classes_file', help='path to the file of all classes with format: cls1,cls2,cls3...',
                        default='app/humanoids_classes.txt')

    parser.add_argument('--image_w', type=int, dest='image_width', help='width of the images',
                        default=320)
    parser.add_argument('--image_h', type=int, dest='image_height', help='height of the images',
                        default=240)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    #initialization
    args = parse_args()

    if not os.path.isfile(args.deeplab_model):
        raise IOError(('Specified model path {:s} not found.\n').format(args.caffemodel))

    #raw_input('press any key to continue')

    detector = Detector(args.in_port_name, args.out_det_img_port_name, args.out_det_port_name, args.rpc_thresh_port_name, args.out_img_port_name, None, args.image_width, args.image_height, args.deeplabmodel, args.cpu_mode, args.gpu_id)

    #raw_input('Detector initialized. \n press any key to continue')

    try:
        # assert yarp.Network.connect(args.out_det_img_port_name, args.viewer_port_name, 'fast_tcp')
        # assert yarp.Network.connect(args.camera_port_name, args.in_port_name, 'fast_tcp')
        detector.run(args.cpu_mode, args.vis)

    finally:
        #        print 'Closing detector'
        detector.cleanup()
