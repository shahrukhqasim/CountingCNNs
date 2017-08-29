import vgg
import tensorflow as tf
from silknet import *
from silknet import LoadInterface
from interface import implements
from silknet import FolderDataReader
import cv2

slim = tf.contrib.slim


class TrainDataLoader(implements(LoadInterface)):
    def __init__(self, image_height, image_width, use_second_frame_instead_of_flow):
        self.image_width = image_width
        self.image_height = image_height
        self.use_second_frame_instead_of_flow = use_second_frame_instead_of_flow

    def load_map(self, path, w, h):
        map = np.fromfile(path, dtype=np.float32)
        print(map)
        return np.reshape(map, (h, w))

    def load_datum(self, full_path):
        indices_x = []
        indices_y = []

        print(full_path)

        image = cv2.imread(os.path.join(full_path, 'frame_1.jpg'))
        if self.use_second_frame_instead_of_flow:
            second_image = cv2.imread(os.path.join(full_path, 'flow.jpg'))
        else:
            second_image = cv2.imread(os.path.join(full_path, 'frame_2.jpg'))

        loaded_image_height, loaded_image_width, _ = np.shape(image)
        density_map = self.load_map(os.path.join(full_path, 'density.dat'), loaded_image_width, loaded_image_height)
        sum = np.sum(density_map)

        image = cv2.resize(image, (self.image_height * 3, self.image_width * 3))
        second_image = cv2.resize(second_image, (self.image_height * 3, self.image_width * 3))
        density_map = cv2.resize(density_map, (int(self.image_height * 3 / 4), int(self.image_width * 3 / 4)))
        sum_2 = np.sum(density_map)
        density_map = (sum / sum_2) * density_map

        input = np.concatenate((image, second_image), axis=2)

        h = self.image_height
        w = self.image_width

        h2 = int(h / 4)
        w2 = int(w / 4)

        inputs = []
        density_maps = []
        print(np.shape(density_map))
        print(h2, w2)
        for i in range(3):
            for j in range(3):
                inputs.append(input[i * h : (i + 1) * h, j * w : (j + 1) * w, :])
                density_maps.append(density_map[i * h2 : (i + 1) * h2, j * w2 : (j + 1) * w2])
                indices_x.append(i)
                indices_y.append(j)

        datum = dict()
        datum['inputs'] = inputs
        datum['density_maps'] = density_maps
        datum['indices_x'] = indices_x
        datum['indices_y'] = indices_y

        return datum

class SimpleCnnNetworkWithFlow:
    def __init__(self):
        self.image_width = 200
        self.image_height = 200
        self.density_map_width = int(self.image_width / 4)
        self.density_map_height = int(self.image_height / 4)
        self.learning_rate = 0.00001
        self.saver_vgg = None
        self.saver_all = None
        self.use_second_frame_instead_of_flow = True
        self.keep_secondary_image_separate = True

        if self.use_second_frame_instead_of_flow:
            self.full_model_path = 'models/model_full_with_two_frames.ckpt'
        else:
            self.full_model_path = 'models/model_full_with_flow.ckpt'

        if self.keep_secondary_image_separate:
            self.full_model_path = 'models/model_full_separate.ckpt'

        self.from_scratch = True
        self.data_path = '/home/srq/Datasets/CUHK Crowd/Dataset_Ready/Train'
        self.test_data_path = '/home/srq/Datasets/CUHK Crowd/Dataset_Ready/Test'
        self.EPOCHS = 50


    def get_r1(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 16, [9, 9], scope='r1_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r1_p1')
            net = slim.conv2d(net, 32, [7, 7], scope='r1_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r1_p2')
            net = slim.conv2d(net, 16, [7, 7], scope='r1_c3')
            net = slim.conv2d(net, 8, [7, 7], scope='r1_c4')
            net = slim.conv2d(net, 1, [1, 1], scope='r1_c5')
        return net


    def get_r1_2(self, x1, x2):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net1 = slim.conv2d(x1, 16, [9, 9], scope='r1_c1_1')
            net1 = slim.max_pool2d(net1, [2, 2], scope='r1_p1_1')
            net1 = slim.conv2d(net1, 32, [7, 7], scope='r1_c2_1')
            net1 = slim.max_pool2d(net1, [2, 2], scope='r1_p2_1')

            net2 = slim.conv2d(x2, 16, [9, 9], scope='r1_c1_2')
            net2 = slim.max_pool2d(net2, [2, 2], scope='r1_p1_2')
            net2 = slim.conv2d(net2, 32, [7, 7], scope='r1_c2_2')
            net2 = slim.max_pool2d(net2, [2, 2], scope='r1_p2_2')

            net = tf.concat([net1, net2], axis=3)

            net = slim.conv2d(net, 16, [7, 7], scope='r1_c3')
            net = slim.conv2d(net, 8, [7, 7], scope='r1_c4')
            net = slim.conv2d(net, 1, [1, 1], scope='r1_c5')

        return net

    # def get_r1(self, x):
    #     with slim.arg_scope(vgg.vgg_arg_scope()):
    #         net = slim.conv2d(x, 24, [5, 5], scope='r3_c1')
    #         net = slim.max_pool2d(net, [2, 2], scope='r3_p1')
    #         net = slim.conv2d(net, 48, [3, 3], scope='r3_c2')
    #         net = slim.max_pool2d(net, [2, 2], scope='r3_p2')
    #         net = slim.conv2d(net, 24, [3, 3], scope='r3_c3')
    #         net = slim.conv2d(net, 12, [3, 3], scope='r3_c4')
    #         net = slim.conv2d(net, 1, [1, 1], scope='r3_c5')
    #     return net
    #
    # def get_r1_2(self, x1, x2):
    #     with slim.arg_scope(vgg.vgg_arg_scope()):
    #         net1 = slim.conv2d(x1, 24, [5, 5], scope='r1_c1_1')
    #         net1 = slim.max_pool2d(net1, [2, 2], scope='r1_p1_1')
    #         net1 = slim.conv2d(net1, 48, [3, 3], scope='r1_c2_1')
    #         net1 = slim.max_pool2d(net1, [2, 2], scope='r1_p2_1')
    #
    #         net2 = slim.conv2d(x2, 24, [5, 5], scope='r1_c1_2')
    #         net2 = slim.max_pool2d(net2, [2, 2], scope='r1_p1_2')
    #         net2 = slim.conv2d(net2, 48, [3, 3], scope='r1_c2_2')
    #         net2 = slim.max_pool2d(net2, [2, 2], scope='r1_p2_2')
    #
    #         net = tf.concat([net1, net2], axis=3)
    #
    #         net = slim.conv2d(net, 24, [3, 3], scope='r3_c3')
    #         net = slim.conv2d(net, 12, [3, 3], scope='r3_c4')
    #         net = slim.conv2d(net, 1, [1, 1], scope='r3_c5')
    #
    #     return net


    def construct_graphs(self):
        if self.keep_secondary_image_separate:
            regressor_input_1 = self.regressor_input_1 = tf.placeholder("float32",
                                                                        shape=[1, self.image_height, self.image_width, 3])
            regressor_input_2 = self.regressor_input_2 = tf.placeholder("float32",
                                                                        shape=[1, self.image_height, self.image_width, 3])
        else:
            regressor_input = self.regressor_input = tf.placeholder("float32", shape=[1, self.image_height, self.image_width, 6])

        regressor_output_ground_truth = self.regressor_output_ground_truth = tf.placeholder("float32",
                                                       shape=[1, self.density_map_height, self.density_map_width])

        with slim.arg_scope(vgg.vgg_arg_scope()):
            if self.keep_secondary_image_separate:
                r1_output = self.get_r1_2(regressor_input_1, regressor_input_2)
            else:
                r1_output = self.get_r1(regressor_input)

            self.cost_r1 = tf.reduce_sum(tf.pow(
                tf.subtract(tf.squeeze(r1_output), tf.scalar_mul(1000, tf.squeeze(regressor_output_ground_truth))), 2))
            self.sum_r1 = tf.scalar_mul(0.001, tf.reduce_sum(r1_output))
            self.optimizer_r1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_r1)

            self.saver_all = tf.train.Saver()

    def run_training(self):
        init = tf.global_variables_initializer()

        dataset = FolderDataReader(self.data_path, TrainDataLoader(self.image_height, self.image_width, self.use_second_frame_instead_of_flow))
        dataset.init()

        with tf.Session() as sess:
            sess.run(init)
            if not self.from_scratch:
                self.saver_all.restore(sess, self.full_model_path)
            # ======================================== Pre-training start ==================================================
            iteration = 0
            print("========== Starting Pre-Training =========")
            while True:
                if dataset.get_next_epoch() == self.EPOCHS:
                    break

                datum, epoch, id = dataset.next_element()
                inputs = datum['inputs']
                density_maps = datum['density_maps']

                assert(len(inputs) == 9 and len(density_maps) == 9)

                for i in range(9):
                    if self.keep_secondary_image_separate:
                        image1 = (inputs[i])[:, :, 0:3]
                        image2 = (inputs[i])[:, :, 3:6]
                    else:
                        image = inputs[i]
                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)
                    # SGD backprop through all of these0

                    if self.keep_secondary_image_separate:
                        c1, s1, o1 = sess.run([self.cost_r1, self.sum_r1, self.optimizer_r1],
                                              feed_dict={self.regressor_input_1: [image1],
                                                         self.regressor_input_2: [image2],
                                                         self.regressor_output_ground_truth: [density_map]})
                    else:
                        c1, s1, o1 = sess.run([self.cost_r1, self.sum_r1, self.optimizer_r1],
                                              feed_dict={self.regressor_input: [image],
                                                         self.regressor_output_ground_truth: [density_map]})
                    print("\tEpoch", epoch, "Iteration", iteration, "Patch", i+1)
                    print("\tCost R1", c1, "Original sum:", sum_gt, "Predicted sum", s1)

                iteration += 1

            self.saver_all.save(sess, self.full_model_path)
            # print("========== Pre-Training Complete =========")
            dataset.halt()

    def run_tests(self):
        init = tf.global_variables_initializer()

        dataset = FolderDataReader(self.test_data_path, TrainDataLoader(self.image_height, self.image_width, self.use_second_frame_instead_of_flow))
        dataset.init()

        total_examples = 0
        total_absolute_error = 0
        total_square_error = 0
        total_gt_sum = 0

        with tf.Session() as sess:
            sess.run(init)
            self.saver_all.restore(sess, self.full_model_path)
            iteration = 0
            while True:
                if dataset.get_next_epoch() == 1:
                    break

                datum, epoch, id = dataset.next_element()
                inputs = datum['inputs']
                density_maps = datum['density_maps']

                assert(len(inputs) == 9 and len(density_maps) == 9)

                sum_gt_total = 0
                sum_predicted_total = 0

                for i in range(9):
                    if self.keep_secondary_image_separate:
                        image1 = (inputs[i])[:, :, 0:3]
                        image2 = (inputs[i])[:, :, 3:6]
                    else:
                        image = inputs[i]

                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)
                    sum_gt_total += sum_gt
                    # SGD backprop through all of these0
                    if self.keep_secondary_image_separate:
                        sum_predicted = sess.run([self.sum_r1], feed_dict={self.regressor_input_1: [image1], self.regressor_input_2: [image2]})
                    else:
                        sum_predicted = sess.run([self.sum_r1], feed_dict={self.regressor_input: [image]})
                    sum_predicted_total += sum_predicted[0]

                total_absolute_error += abs(sum_predicted_total - sum_gt_total)
                total_square_error += pow(abs(sum_predicted_total - sum_gt_total), 2)
                total_gt_sum += sum_gt_total

                iteration += 1
                total_examples += 1

            dataset.halt()

        mean_absolute_error = total_absolute_error / total_examples
        mean_squared_error = np.sqrt(total_square_error) / total_examples
        mean_people_per_image = total_gt_sum / total_examples

        print("MAE", mean_absolute_error)
        print("MSE", mean_squared_error)
        print("Mean people per image", mean_people_per_image)
