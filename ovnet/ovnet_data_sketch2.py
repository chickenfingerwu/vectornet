# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import io
from random import shuffle
import xml.etree.ElementTree as et
import copy
import multiprocessing.managers
import multiprocessing.pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/sketch_schneider',
                           """Path to the Sketch data directory.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")
tf.app.flags.DEFINE_integer('stroke_width', 10,
                            """stroke width.""")

class MPManager(multiprocessing.managers.SyncManager):
    pass
MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)


class Param(object):
    def __init__(self):
        self.image_width = FLAGS.image_width
        self.image_height = FLAGS.image_height
        self.use_two_channels = FLAGS.use_two_channels
        self.stroke_width = FLAGS.stroke_width

            
class BatchManager(object):
    def __init__(self):
        # read all svg files
        self._next_svg_id = 0
        self._svg_list = []
        if FLAGS.file_list:
            file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self._svg_list.append(file_path)
                    # with open(file_path, 'r') as sf:
                    #     svg = sf.read()
                    #     self._svg_list.append(svg)

        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('svg_pre'):
                        continue

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        svg = f.read()
                        self._svg_list.append(svg)

        self.num_examples_per_epoch = len(self._svg_list)
        self.num_epoch = 1

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size

        if FLAGS.num_processors == 1:
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        else:
            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self._svg_batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, svg_batch=self._svg_batch, x_batch=self.x_batch, y_batch=self.y_batch, FLAGS=Param())

    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            svg_batch = []
            for i in xrange(FLAGS.batch_size):
                svg_batch.append(self._svg_list[self._next_svg_id])
                train_set(i, svg_batch, self.x_batch, self.y_batch, FLAGS)
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._svg_list)
        else:
            for i in xrange(FLAGS.batch_size):
                self._svg_batch[i] = self._svg_list[self._next_svg_id]
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._svg_list)

            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.x_batch, self.y_batch


def train_set(batch_id, svg_batch, x_batch, y_batch, FLAGS):
    with open(svg_batch[batch_id], 'r') as sf:
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height, sw=FLAGS.stroke_width,
                               bx=0, by=0, bw=800, bh=800)

    x_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    x_img = Image.open(io.BytesIO(x_png))
    x = np.array(x_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(x)
    x = x / max_intensity    

    # # debug
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.show()

    y = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
    stroke_list = []
    svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in xrange(num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[0][i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[0][c] != stroke:
                svg_xml[0].remove(svg_xml[0][c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)

    for i in xrange(num_paths-1):
        for j in xrange(i+1, num_paths):
            intersect = np.logical_and(stroke_list[i], stroke_list[j])
            y = np.logical_or(intersect, y)

            # # debug
            # plt.figure()
            # plt.subplot(221)
            # plt.imshow(stroke_list[i], cmap=plt.cm.gray)
            # plt.subplot(222)
            # plt.imshow(stroke_list[j], cmap=plt.cm.gray)
            # plt.subplot(223)
            # plt.imshow(intersect, cmap=plt.cm.gray)
            # plt.subplot(224)
            # plt.imshow(y, cmap=plt.cm.gray)
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            # plt.show()
    
    # y = np.multiply(x, y) * 1000
    # y = y.astype(np.float) * 1000

    # # debug
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.subplot(122)
    # plt.imshow(y, cmap=plt.cm.gray)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()

    x_batch[batch_id,:,:,0] = x
    y_batch[batch_id,:,:,0] = y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('ovnet'):
        working_path = os.path.join(current_path, 'vectornet/ovnet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_processors = 1

    batch_manager = BatchManager()
    x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

    print('Done')