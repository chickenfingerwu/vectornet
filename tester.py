from __future__ import print_function

from glob import glob
from cv2 import cv2

import imageio
import io

import os
import multiprocessing
import time
from datetime import datetime
import platform
from subprocess import call
from shutil import copyfile
from preprocess import thinning
import re

import cairosvg
from PIL import Image
import sklearn.neighbors
import skimage.measure

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from models import *
from utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def vectorize_mp(q, IoU_q):
    while True:
        pm = q.get()
        if pm is None:
            break
        pm = vectorize(pm, False, '')
        IoU_q.put(pm.acc_avg)
        # print('%s: qsize %d' % (datetime.now(), q.qsize()))
        q.task_done()


def read_png(file_path, img_domain, is_gt):
    img = thinning(file_path, img_domain, is_gt)
    # img = Image.fromarray(img).convert('L')
    # img = ImageOps.invert(img)
    s = np.array(img)[:, :].astype(np.float)  # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity
    return s


def read_svg(file_path, width, height):
    with open(file_path, 'r', encoding='utf-8') as f:
        svg = f.read()

    r = 0
    s = [1, 1]
    t = [0, 0]
    # if transform:
    #     r = rng.randint(-45, 45)
    #     # s_sign = rng.choice([1, -1], 1)[0]
    #     s_sign = 1
    #     s = 1.75 * rng.random_sample(2) + 0.25 # [0.25, 2)
    #     s[1] = s[1] * s_sign
    #     t = rng.randint(-10, 10, 2)
    #     if s_sign == -1:
    #         t[1] = t[1] - 109

    path_list = []
    pid = 0
    num_paths = 0
    while pid != -1:
        pid = svg.find('path id', pid + 1)
        num_paths = num_paths + 1
    num_paths = num_paths - 1  # uncount last one

    for i in range(num_paths):
        svg_one = svg
        pid = len(svg_one)
        for j in range(num_paths):
            pid = svg_one.rfind('path id', 0, pid)
            if j != i:
                id_start = svg_one.rfind('>', 0, pid) + 1
                id_end = svg_one.find('/>', id_start) + 2
                svg_one = svg_one[:id_start] + svg_one[id_end:]

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
        y_img = Image.open(io.BytesIO(y_png))
        y_img = y_img.resize((width, height))
        path = (np.array(y_img)[:, :, 3] > 0)
        path_list.append(path)

    return num_paths, path_list


class Param(object):
    pass


class Tester(object):
    def __init__(self, config):
        tf.set_random_seed(config.random_seed)
        self.config = config
        # self.batch_manager = batch_manager
        self.rng = np.random.RandomState(config.random_seed)
        self.data_paths_A = sorted(glob("{}/trainA/*.{}".format(self.config.data_dir, 'png')))
        self.data_paths_B = sorted(glob("{}/trainB/*.{}".format(self.config.data_dir, 'png')))

        self.b_num = config.test_batch_size
        self.height = config.height
        self.width = config.width
        self.conv_hidden_num = config.conv_hidden_num
        self.repeat_num = config.repeat_num
        self.data_format = config.data_format
        self.use_norm = config.use_norm

        self.load_pathnet = config.load_pathnet
        self.load_overlapnet = config.load_overlapnet
        self.find_overlap = config.find_overlap
        self.overlap_threshold = config.overlap_threshold
        self.max_label = config.max_label
        self.label_cost = config.label_cost
        self.sigma_neighbor = config.sigma_neighbor
        self.sigma_predict = config.sigma_predict
        self.neighbor_sample = config.neighbor_sample
        self.IoU_list = []

        self.num_test = config.num_test
        # self.test_paths = self.batch_manager.test_paths
        # if config.dataset == 'baseball' or config.dataset == 'cat' or \
        #         config.dataset == 'multi':
        #     self.test_paths = self.batch_manager.vec_paths
        # if self.num_test < len(self.test_paths):
        #     self.test_paths = self.rng.choice(self.test_paths, self.num_test, replace=False)
        self.mp = config.mp
        self.num_worker = config.num_worker

        self.model_dir = config.model_dir
        self.data_path = config.data_path
        self.build_model()

    def build_model(self):
        pathnet_graph = tf.Graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True))
        self.sp = tf.Session(config=sess_config, graph=pathnet_graph)
        with pathnet_graph.as_default():
            self.xp = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 2])
            if self.data_format == 'NCHW':
                self.xp = nhwc_to_nchw(self.xp)

            self.yp, _ = VDSR(self.xp, self.conv_hidden_num, self.repeat_num,
                              self.data_format, self.use_norm, train=False)
            # show_all_variables()

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.load_pathnet)
            assert (ckpt and self.load_pathnet)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sp, os.path.join(self.load_pathnet, ckpt_name))
            print('%s: Pre-trained model restored from %s' % (datetime.now(), self.load_pathnet))

        if self.find_overlap:
            overlapnet_graph = tf.Graph()
            self.so = tf.Session(config=sess_config, graph=overlapnet_graph)
            with overlapnet_graph.as_default():
                self.xo = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1])
                if self.data_format == 'NCHW':
                    self.xo = nhwc_to_nchw(self.xo)

                self.yo, _ = VDSR(self.xo, self.conv_hidden_num, self.repeat_num,
                                  self.data_format, self.use_norm, train=False)
                show_all_variables()

                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.load_overlapnet)
                assert (ckpt and self.load_overlapnet)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(self.so, os.path.join(self.load_overlapnet, ckpt_name))
                print('%s: Pre-trained model restored from %s' % (datetime.now(), self.load_overlapnet))

    def test(self):
        if self.mp:
            IoU_q = multiprocessing.Queue()
            q = multiprocessing.JoinableQueue()
            pool = multiprocessing.Pool(self.num_worker, vectorize_mp, (q, IoU_q))
        # file_path = self.config.file_path
        # param = self.predict(file_path)
        # if self.mp:
        #     q.put(param)
        # else:
        #     vectorize(param)

        # preprocess first
        num_data_A = len(self.data_paths_A)
        self.current_data_path = self.config.data_dir + '/trainA'
        for i in range(num_data_A):
            file_path = self.data_paths_A[i]
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            if not os.path.isfile(os.path.join(self.current_data_path + '/svg/', '%s_gr_truth.svg' % (file_name))):
                img = cv2.imread(file_path)
                print('\n[{}/{}] start prediction, path: {}'.format(i + 1, num_data_A, file_path))

                param = self.predict(img, file_name, 'A', False)

                if self.mp:
                    q.put(param)
                else:
                    vectorize(param, self.width, self.height)

        if self.mp:
            q.join()
            pool.terminate()
            pool.join()

        if self.mp:
            IoU_q = multiprocessing.Queue()
            q = multiprocessing.JoinableQueue()
            pool = multiprocessing.Pool(self.num_worker, vectorize_mp, (q, IoU_q))

        num_data_B = len(self.data_paths_B)
        self.current_data_path = self.config.data_dir + '/trainB'
        for i in range(num_data_B):
            file_path = self.data_paths_B[i]
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            if not os.path.isfile(os.path.join(self.current_data_path + '/svg/', '%s_gr_truth.svg' % (file_name))):
                img = cv2.imread(file_path)
                print('\n[{}/{}] start prediction, path: {}'.format(i + 1, num_data_A, file_path))

                param = self.predict(img, file_name, 'B', False)

                if self.mp:
                    q.put(param)
                else:
                    vectorize(param, self.width, self.height)

        if self.mp:
            q.join()
            pool.terminate()
            pool.join()

        # self.stat()

    # def start_process_for_compute_IoU_loss(self, real_images, fake_images, file_names, direction):
    #     p = multiprocessing.Process(target=self.compute_loss_structure, args=(real_images, fake_images, file_names, direction,))
    #     p.start()
    #     p.join()
    #     return self.IoU_acc

    def compute_loss_structure(self, real_images, fake_images, file_names, direction, img_domain):
        if self.mp:
            q = multiprocessing.JoinableQueue()
            IoU_q = multiprocessing.Queue()
            pool = multiprocessing.Pool(self.num_worker, vectorize_mp, (q, IoU_q))
        bs = real_images.shape[0]
        IoU_list = []
        for i in range(bs):
            real_img = real_images[i]
            fake_img = fake_images[i]
            file_name = os.path.splitext(os.path.basename(file_names[i]))[0]
            if self.mp:
                real_pm = self.predict(real_img, file_name + '_real_' + img_domain + '_' + direction, img_domain, False)
                fake_pm = self.predict(fake_img, file_name + '_fake_' + img_domain + '_' + direction, img_domain, True)
                q.put([real_pm, fake_pm, file_name])
            else:
                real_pm = self.predict(real_img, file_name + '_real_' + img_domain + '_' + direction, img_domain, False)
                fake_pm = self.predict(fake_img, file_name + '_fake_' + img_domain + '_' + direction, img_domain, True)
                real_pm = vectorize(real_pm, False, '')
                fake_pm = vectorize(fake_pm, True, real_pm.svg_path)
                IoU_list.append(fake_pm.acc_avg)
        if self.mp:
            q.join()
            pool.terminate()
            pool.join()
            while not IoU_q.empty():
                IoU_list.append(IoU_q.get())
        print('all acc: ', IoU_list)
        self.IoU_acc = np.average(IoU_list)
        return self.IoU_acc

    def predict(self, img, file_name, img_domain, has_ground_truth):
        # convert svg to raster image
        img = read_png(img, img_domain, not has_ground_truth)
        # file_name = os.path.splitext(os.path.basename(file_path))[0]
        input_img_path = os.path.join(self.model_dir, '%s_0_input.png' % file_name)
        save_image((1 - img[np.newaxis, :, :, np.newaxis]) * 255, input_img_path, padding=0)

        # # debug
        # print(num_paths)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()

        pm = Param()

        # predict paths through pathnet
        start_time = time.time()
        paths, path_pixels = self.extract_path(img)
        num_path_pixels = len(path_pixels[0])
        pids = self.rng.randint(num_path_pixels, size=8)
        path_img_path = os.path.join(self.model_dir, '%s_1_path.png' % file_name)
        save_image((1 - paths[pids, :, :, :]) * 255, path_img_path, padding=0)

        # # debug
        # plt.imshow(paths[0,:,:,0], cmap=plt.cm.gray)
        # plt.show()

        duration = time.time() - start_time
        print('%s: %s, predict paths (#pixels:%d) through pathnet (%.3f sec)' % (
            datetime.now(), file_name, num_path_pixels, duration))
        pm.duration_pred = duration
        pm.duration = duration

        dup_dict = {}
        dup_rev_dict = {}
        dup_id = num_path_pixels  # start id of duplicated pixels

        if self.find_overlap:
            # predict overlap using overlap net
            start_time = time.time()
            ov = self.overlap(img)

            overlap_img_path = os.path.join(self.model_dir, '%s_2_overlap.png' % file_name)
            ov_img = ov[np.newaxis, :, :, np.newaxis]
            save_image((1 - ov_img) * 255, overlap_img_path, padding=0)

            # # debug
            # plt.imshow(ov, cmap=plt.cm.gray)
            # plt.show()

            for i in range(num_path_pixels):
                if ov[path_pixels[0][i], path_pixels[1][i]]:
                    dup_dict[i] = dup_id
                    dup_rev_dict[dup_id] = i
                    dup_id += 1

            # debug
            # print(dup_dict)
            # print(dup_rev_dict)

            duration = time.time() - start_time
            print('%s: %s, predict overlap (#:%d) through ovnet (%.3f sec)' % (
                datetime.now(), file_name, dup_id - num_path_pixels, duration))
            pm.duration_ov = duration
            pm.duration += duration
        else:
            pm.duration_ov = 0

        # write config file for graphcut
        start_time = time.time()
        tmp_dir = os.path.join(self.model_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        pred_file_path = os.path.join(tmp_dir, file_name + '.pred')
        f = open(pred_file_path, 'w')
        # info
        f.write(pred_file_path + '\n')
        f.write(self.data_path + '\n')
        f.write('%d\n' % self.max_label)
        f.write('%d\n' % self.label_cost)
        f.write('%f\n' % self.sigma_neighbor)
        f.write('%f\n' % self.sigma_predict)
        # f.write('%d\n' % num_path_pixels)
        f.write('%d\n' % dup_id)

        # support only symmetric edge weight
        radius = self.sigma_neighbor * 2
        nb = sklearn.neighbors.NearestNeighbors(radius=radius)
        nb.fit(np.array(path_pixels).transpose())

        high_spatial = 100000
        for i in range(num_path_pixels - 1):
            p1 = np.array([path_pixels[0][i], path_pixels[1][i]])
            pred_p1 = np.reshape(paths[i, :, :, :], [self.height, self.width])

            # see close neighbors and some far neighbors (stochastic sampling)
            rng = nb.radius_neighbors([p1])
            num_close = len(rng[1][0])
            far = np.setdiff1d(range(i + 1, num_path_pixels), rng[1][0])
            num_far = len(far)
            num_far = int(num_far * self.neighbor_sample)
            if num_far > 0:
                far_ids = self.rng.choice(far, size=num_far)
                nb_ids = np.concatenate((rng[1][0], far_ids))
            else:
                nb_ids = rng[1][0]

            for rj, j in enumerate(nb_ids):  # ids
                if j <= i:
                    continue
                p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
                if rj < num_close:
                    d12 = rng[0][0][rj]
                else:
                    d12 = np.linalg.norm(p1 - p2, 2)

                # for j in range(i+1, num_path_pixels): # see entire neighbors
                #     p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
                #     d12 = np.linalg.norm(p1-p2, 2)

                pred_p2 = np.reshape(paths[j, :, :, :], [self.height, self.width])
                pred = (pred_p1[p2[0], p2[1]] + pred_p2[p1[0], p1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0 - pred) ** 2 / self.sigma_predict ** 2)

                spatial = np.exp(-0.5 * d12 ** 2 / self.sigma_neighbor ** 2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))

                dup_i = dup_dict.get(i)
                if dup_i is not None:
                    f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial))  # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (i, dup_i, 0, high_spatial))  # shouldn't be labeled together
                dup_j = dup_dict.get(j)
                if dup_j is not None:
                    f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial))  # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (j, dup_j, 0, high_spatial))  # shouldn't be labeled together

                if dup_i is not None and dup_j is not None:
                    f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial))  # dup_i < dup_j

        f.close()
        duration = time.time() - start_time
        # print('%s: %s, prediction computed (%.3f sec)' % (datetime.now(), file_name, duration))
        pm.duration_map = duration
        pm.duration += duration

        pm.path_pixels = path_pixels
        pm.dup_dict = dup_dict
        pm.dup_rev_dict = dup_rev_dict
        pm.img = img
        pm.file_name = file_name
        # pm.file_path = file_path
        pm.model_dir = self.model_dir
        pm.height = self.height
        pm.width = self.width
        pm.max_label = self.max_label
        pm.sigma_neighbor = self.sigma_neighbor
        pm.sigma_predict = self.sigma_predict
        pm.is_gt = has_ground_truth

        pm.data_path = self.current_data_path

        return pm

    def extract_path(self, img):
        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0])
        assert (num_path_pixels > 0)

        y_batch = None
        for b in range(0, num_path_pixels, self.b_num):
            b_size = min(self.b_num, num_path_pixels - b)
            x_batch = np.zeros([b_size, self.height, self.width, 2])
            for i in range(b_size):
                x_batch[i, :, :, 0] = img
                px, py = path_pixels[0][b + i], path_pixels[1][b + i]
                x_batch[i, px, py, 1] = 1.0

            if self.data_format == 'NCHW':
                x_batch = to_nchw_numpy(x_batch)
            y_b = self.sp.run(self.yp, feed_dict={self.xp: x_batch})
            y_b = np.clip(y_b, 0, 1)
            if self.data_format == 'NCHW':
                y_b = to_nhwc_numpy(y_b)
            if y_batch is None:
                y_batch = y_b
            else:
                y_batch = np.concatenate((y_batch, y_b), axis=0)

        return y_batch, path_pixels

    def overlap(self, img):
        x_batch = np.zeros([1, self.height, self.width, 1])
        x_batch[0, :, :, 0] = img

        if self.data_format == 'NCHW':
            x_batch = to_nchw_numpy(x_batch)
        y_b = self.so.run(self.yo, feed_dict={self.xo: x_batch})
        if self.data_format == 'NCHW':
            y_b = to_nhwc_numpy(y_b)
        return (y_b[0, :, :, 0] >= self.overlap_threshold)

    def stat(self):
        from glob import glob
        stat_paths = sorted(glob("{}/*{}".format(self.model_dir, '_stat.txt')))
        diff = []
        abs_diff = []
        acc = []
        d_pred = []
        d_ov = []
        d_map = []
        d_vec = []
        duration = []

        # print(len(stat_paths))
        for path in stat_paths:
            with open(path, 'r') as f:
                stat = f.readline()
                # print(stat)
            stat = stat.split()
            # file_path, num_labels, pm.num_paths, acc_avg,
            # duration_pred, duration_ov, duration_map,
            # duration_vect, duration
            num_labels = int(stat[1])
            gt_labels = int(stat[2])
            acc_ = float(stat[3])
            dpred = float(stat[4])
            dov = float(stat[5])
            dmap = float(stat[6])
            dvec = float(stat[7])
            d = float(stat[8])

            diff.append(num_labels - gt_labels)
            abs_diff.append(abs(num_labels - gt_labels))
            acc.append(acc_)
            d_pred.append(dpred)
            d_ov.append(dov)
            d_map.append(dmap)
            d_vec.append(dvec)
            duration.append(d)

        print('label abs diff: {}'.format(np.average(abs_diff)))
        print('acc: {}'.format(np.average(acc)))
        print('duration for prediction: {}'.format(np.average(d_pred)))
        print('duration for overlap: {}'.format(np.average(d_ov)))
        print('duration for mapping: {}'.format(np.average(d_map)))
        print('duration for vectorization: {}'.format(np.average(d_vec)))
        print('duration total: {}'.format(np.average(duration)))

        stat_path = os.path.join(self.model_dir, 'summary.txt')
        with open(stat_path, 'w') as f:
            f.write('label abs diff: {}\n'.format(np.average(abs_diff)))
            f.write('acc: {}\n'.format(np.average(acc)))
            f.write('duration for prediction: {}\n'.format(np.average(d_pred)))
            f.write('duration for overlap: {}\n'.format(np.average(d_ov)))
            f.write('duration for mapping: {}\n'.format(np.average(d_map)))
            f.write('duration for vectorization: {}\n'.format(np.average(d_vec)))
            f.write('duration total: {}\n'.format(np.average(duration)))


def vectorize(pm, do_compute_acc, gt_path):
    start_time = time.time()
    # file_path = os.path.basename(pm.file_path)
    file_name = pm.file_name

    num_paths, path_list = 0, []
    if do_compute_acc:
        num_paths, path_list = read_svg(gt_path, pm.width, pm.height)
    pm.num_paths = num_paths
    pm.path_list = path_list

    # 1. label
    labels, e_before, e_after = label(file_name, pm)

    # 2. merge small components
    labels = merge_small_component(labels, pm)

    # # 2-2. assign one label per one connected component
    # labels = label_cc(labels, pm)

    # 3. compute accuracy
    unique_labels = np.unique(labels)
    num_labels = unique_labels.size
    acc_avg = 0

    has_ground_truth = do_compute_acc
    if has_ground_truth:
        accuracy_list = compute_accuracy(labels, pm)
        acc_avg = np.average(accuracy_list)

        # print('%s: %s, the number of labels %d, truth %d' % (datetime.now(), file_name, num_labels, pm.num_paths))
        # print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
        # print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
        # print('%s: %s, accuracy computed, avg.: %.3f' % (datetime.now(), file_name, acc_avg))
    pm.acc_avg = acc_avg

    # 4. save image
    if not has_ground_truth:
        pm.svg_path = save_label_img(labels, unique_labels, num_labels, 0, pm)
        duration = time.time() - start_time
        pm.duration_vect = duration
        pm.duration += duration

    ## write result
    # print('%s: %s, done (%.3f sec)' % (datetime.now(), file_name, pm.duration))
    # stat_file_path = os.path.join(pm.model_dir, file_name + '_stat.txt')
    # with open(stat_file_path, 'w') as f:
    #     f.write('%s %d %d %.3f %.3f %.3f %.3f %.3f %.3f\n' % (
    #         file_path, num_labels, pm.num_paths, acc_avg,
    #         pm.duration_pred, pm.duration_ov, pm.duration_map,
    #         pm.duration_vect, pm.duration))
    return pm


def label(file_name, pm):
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/build')
    os.chdir(gco_path)

    pred_file_path = os.path.join(working_path, pm.model_dir, 'tmp', file_name + '.pred')
    sys_name = platform.system()
    if sys_name == 'Windows':
        call(['Release/gco.exe', pred_file_path])
    else:
        call(['./gco', pred_file_path])
    os.chdir(working_path)

    # read graphcut result
    label_file_path = os.path.join(pm.model_dir, 'tmp', file_name + '.label')
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))

    return labels, e_before, e_after


def merge_small_component(labels, pm):
    knb = sklearn.neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knb.fit(np.array(pm.path_pixels).transpose())

    num_path_pixels = len(pm.path_pixels[0])

    for iter in range(2):
        # # debug
        # print('%d-th iter' % iter)

        unique_label = np.unique(labels)
        for i in unique_label:
            i_label_list = np.nonzero(labels == i)

            # handle duplicated pixels
            for j, i_label in enumerate(i_label_list[0]):
                if i_label >= num_path_pixels:
                    i_label_list[0][j] = pm.dup_rev_dict[i_label]

            # connected component analysis on 'i' label map
            i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
            i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = 1.0
            cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

            # # debug
            # print('%d: # labels %d, # cc %d' % (i, num_i_label_pixels, num_cc))
            # plt.imshow(cc_map, cmap='spectral')
            # plt.show()

            # detect small pixel component
            for j in range(num_cc):
                j_cc_list = np.nonzero(cc_map == (j + 1))
                num_j_cc = len(j_cc_list[0])

                # consider only less than 5 pixels component
                if num_j_cc > 4:
                    continue

                # assign dominant label of neighbors using knn
                for k in range(num_j_cc):
                    p1 = np.array([j_cc_list[0][k], j_cc_list[1][k]])
                    _, indices = knb.kneighbors([p1], n_neighbors=5)
                    max_label_nb = np.argmax(np.bincount(labels[indices][0]))
                    labels[indices[0][0]] = max_label_nb

                    # # debug
                    # print(' (%d,%d) %d -> %d' % (p1[0], p1[1], i, max_label_nb))

                    dup = pm.dup_dict.get(indices[0][0])
                    if dup is not None:
                        labels[dup] = max_label_nb

    return labels


def label_cc(labels, pm):
    unique_label = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    new_label = pm.max_label
    for i in unique_label:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        # connected component analysis on 'i' label map
        i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = 1.0
        cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

        if num_cc > 1:
            for i_label in i_label_list[0]:
                cc_label = cc_map[pm.path_pixels[0][i_label], pm.path_pixels[1][i_label]]
                if cc_label > 1:
                    labels[i_label] = new_label + (cc_label - 2)

            new_label += (num_cc - 1)

    return labels


def compute_accuracy(labels, pm):
    unique_labels = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])
    acc_id_list = []
    acc_list = []
    for i in unique_labels:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        i_label_map = np.zeros([pm.height, pm.width], dtype=np.bool)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = True

        accuracy_list = []
        for j, stroke in enumerate(pm.path_list):
            # accuracy = np.sum(np.logical_xor(i_label_map, stroke) ** 2)
            intersect = np.sum(np.logical_and(i_label_map, stroke))
            union = np.sum(np.logical_or(i_label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' %
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        # if acc > 0.1:
        acc_id_list.append(id)
        acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return acc_list


def save_label_img(labels, unique_labels, num_labels, acc_avg, pm):
    sys_name = platform.system()

    # file_path = os.path.basename(pm.file_path)
    file_name = pm.file_name
    num_path_pixels = len(pm.path_pixels[0])
    gt_labels = 0
    if pm.is_gt:
        gt_labels = pm.num_paths

    cmap = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=num_labels - 1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
    label_map_t = np.ones([pm.height, pm.width, 3], dtype=np.float)
    first_svg = True
    target_svg_path = os.path.join(pm.data_path + '/svg/', '%s_gr_truth.svg' % (file_name))
    gid, pid = 0, 0
    for color_id, i in enumerate(unique_labels):
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        color = np.asarray(cscalarmap.to_rgba(color_id))
        label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = color[:3]

        # save i label map
        i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = pm.img[
            pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]]
        _, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)
        i_label_map_path = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.bmp' % (file_name, i, num_cc))
        imageio.imwrite(i_label_map_path, i_label_map)

        i_label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = color[:3]
        label_map_t += i_label_map

        # vectorize using potrace
        color *= 255
        color_hex = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

        if sys_name == 'Windows':
            potrace_path = os.path.join('potrace', 'potrace.exe')
            call([potrace_path, '-s', '-i', '-C' + color_hex, i_label_map_path])
        else:
            call(['potrace', '-s', '-i', '-C' + color_hex, i_label_map_path])

        i_label_map_svg = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.svg' % (file_name, i, num_cc))
        if first_svg:
            copyfile(i_label_map_svg, target_svg_path)
            first_svg = False
            with open(target_svg_path, 'r') as f:
                target_svg = f.read()

            source_svg = target_svg

            path_start = target_svg.find('<g')
            path_end = target_svg.find('</svg>')

            insert_pos = target_svg.find('</svg>')
            stroke_indexes = [m.start() for m in re.finditer('<path', source_svg)]
            new_source_svg = ''

            if len(stroke_indexes) == 1:
                gid += 1
                pid += 1
                stroke_start = stroke_indexes[0] + 5
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:stroke_start] + ' id="%s-s%d" ' % (
                                     file_name, pid) + source_svg[stroke_start + 1:path_end]
                target_svg = target_svg[:path_start] + new_source_svg + target_svg[insert_pos:]

                with open(target_svg_path, 'w') as f:
                    f.write(target_svg)

            elif len(stroke_indexes) > 1:
                gid += 1
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:stroke_indexes[0] + 5]
                for i_s in range(0, len(stroke_indexes)):
                    pid += 1
                    stroke_start = stroke_indexes[i_s]
                    stroke_start += 5
                    if i_s < len(stroke_indexes) - 1:
                        next_stroke_start = stroke_indexes[i_s + 1] + 5
                    else:
                        next_stroke_start = path_end
                    new_source_svg = new_source_svg + ' id="%s-s%d" ' % (
                        file_name, pid) + source_svg[stroke_start:next_stroke_start]

                target_svg = target_svg[:path_start] + new_source_svg + target_svg[insert_pos:]

                with open(target_svg_path, 'w') as f:
                    f.write(target_svg)
            elif len(stroke_indexes) == 0:
                gid += 1
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:path_end]
                target_svg = target_svg[:path_start] + new_source_svg + target_svg[insert_pos:]
                with open(target_svg_path, 'w') as f:
                    f.write(target_svg)
        else:
            with open(target_svg_path, 'r') as f:
                target_svg = f.read()

            with open(i_label_map_svg, 'r') as f:
                source_svg = f.read()

            path_start = source_svg.find('<g')
            path_end = source_svg.find('</svg>')
            new_source_svg = ''
            stroke_indexes = [m.start() for m in re.finditer('<path', source_svg)]
            insert_pos = target_svg.find('</svg>')
            if len(stroke_indexes) == 1:
                gid += 1
                pid += 1
                stroke_start = stroke_indexes[0] + 5
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:stroke_start] + ' id="%s-s%d" ' % (
                                     file_name, pid) + source_svg[stroke_start + 1:path_end]
            elif len(stroke_indexes) > 1:
                gid += 1
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:stroke_indexes[0] + 5]
                for i_s in range(0, len(stroke_indexes)):
                    pid += 1
                    stroke_start = stroke_indexes[i_s]
                    stroke_start += 5
                    if i_s < len(stroke_indexes) - 1:
                        next_stroke_start = stroke_indexes[i_s + 1] + 5
                    else:
                        next_stroke_start = path_end
                    new_source_svg = new_source_svg + ' id="%s-s%d" ' % (
                        file_name, pid) + source_svg[stroke_start:next_stroke_start]
            elif len(stroke_indexes) == 0:
                gid += 1
                new_source_svg = source_svg[path_start:path_start + 2] + ' id="%s-g%d" ' % (
                    file_name, gid) + source_svg[path_start + 3:path_end]
            target_svg = target_svg[:insert_pos] + new_source_svg + target_svg[insert_pos:]

            with open(target_svg_path, 'w') as f:
                f.write(target_svg)

        # remove i label map
        os.remove(i_label_map_path)
        os.remove(i_label_map_svg)

    # set opacity 0.5 to see overlaps
    with open(target_svg_path, 'r') as f:
        target_svg = f.read()

    insert_pos = target_svg.find('<g')
    target_svg = target_svg[:insert_pos] + '<g fill-opacity="0.5">' + target_svg[insert_pos:]
    insert_pos = target_svg.find('</svg>')
    target_svg = target_svg[:insert_pos] + '</g>' + target_svg[insert_pos:]

    with open(target_svg_path, 'w') as f:
        f.write(target_svg)

    label_map_path = os.path.join(pm.data_path + '/svg/', '%s_vectorized.png' % (file_name))
    imageio.imwrite(label_map_path, label_map)

    # label_map_t /= np.amax(label_map_t)
    # label_map_path = os.path.join(pm.data_path + '/svg/', '%s_%.2f_%.2f_%d_%d_t.png' % (
    #     file_name, pm.sigma_neighbor, pm.sigma_predict, num_labels, gt_labels))
    # imageio.imwrite(label_map_path, label_map_t)
    return target_svg_path
