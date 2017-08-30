import vgg
import tensorflow as tf
from silknet import *
from silknet import LoadInterface
from interface import implements
from silknet import FolderDataReader
import cv2
import matplotlib.pyplot as plt

slim = tf.contrib.slim


class TrainDataLoader(implements(LoadInterface)):
    def __init__(self, image_height, image_width):
        self.image_width = image_width
        self.image_height = image_height

    def load_map(self, path, w, h):
        map = np.fromfile(path, dtype=np.float32)
        print(map)
        return np.reshape(map, (h, w))

    def load_datum(self, full_path):
        images = []
        density_maps = []
        indices_x = []
        indices_y = []

        print(full_path)

        image_full = image = cv2.imread(os.path.join(full_path, 'frame_1.jpg'))
        loaded_image_height, loaded_image_width, _ = np.shape(image)
        density_map = self.load_map(os.path.join(full_path, 'density.dat'), loaded_image_width, loaded_image_height)
        sum = np.sum(density_map)

        image = cv2.resize(image, (self.image_height * 3, self.image_width * 3))
        density_map = cv2.resize(density_map, (int(self.image_height * 3 / 4), int(self.image_width * 3 / 4)))
        sum_2 = np.sum(density_map)
        density_map = (sum / sum_2) * density_map

        h = self.image_height
        w = self.image_width

        h2 = int(h / 4)
        w2 = int(w / 4)

        images = []
        density_maps = []
        print(np.shape(density_map))
        print(h2, w2)
        for i in range(3):
            for j in range(3):
                images.append(image[i * h : (i + 1) * h, j * w : (j + 1) * w, :])
                density_maps.append(density_map[i * h2 : (i + 1) * h2, j * w2 : (j + 1) * w2])
                indices_x.append(i)
                indices_y.append(j)

        datum = dict()
        datum['complete_image'] = image_full
        datum['images'] = images
        datum['density_maps'] = density_maps
        datum['indices_x'] = indices_x
        datum['indices_y'] = indices_y

        return datum


class McnnNetwork:
    def __init__(self):
        self.image_width = 200
        self.image_height = 200
        self.density_map_width = int(self.image_width / 4)
        self.density_map_height = int(self.image_height / 4)
        self.learning_rate = 0.00001
        self.saver_vgg = None
        self.saver_all = None
        self.full_model_path = 'models/model_mcnn.ckpt'
        self.from_scratch = True
        self.data_path = '/home/srq/Datasets/ShanghaiTech/part_A_2/Training'
        self.test_data_path = '/home/srq/Datasets/ShanghaiTech/part_A_2/Testing'
        self.EPOCHS = 80
        self.PRETRAIN_EACH_EPOCHS = 10

    def get_r1(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 16, [9, 9], scope='r1_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r1_p1')
            net = slim.conv2d(net, 32, [7, 7], scope='r1_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r1_p2')
            net = slim.conv2d(net, 16, [7, 7], scope='r1_c3')
            net = slim.conv2d(net, 8, [7, 7], scope='r1_c4')
        return net

    def get_r2(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 20, [7, 7], scope='r2_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r2_p1')
            net = slim.conv2d(net, 40, [5, 5], scope='r2_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r2_p2')
            net = slim.conv2d(net, 20, [5, 5], scope='r2_c3')
            net = slim.conv2d(net, 10, [5, 5], scope='r2_c4')
        return net

    def get_r3(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 24, [5, 5], scope='r3_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r3_p1')
            net = slim.conv2d(net, 48, [3, 3], scope='r3_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r3_p2')
            net = slim.conv2d(net, 24, [3, 3], scope='r3_c3')
            net = slim.conv2d(net,  12, [3, 3], scope='r3_c4')
        return net

    def construct_graphs(self):
        self.r1_out_multiplier = tf.placeholder(tf.float32, ())
        self.r2_out_multiplier = tf.placeholder(tf.float32, ())
        self.r3_out_multiplier = tf.placeholder(tf.float32, ())
        regressor_input = self.regressor_input = tf.placeholder("float32", shape=[1, self.image_height, self.image_width, 3])
        regressor_output_ground_truth = self.regressor_output_ground_truth = tf.placeholder("float32",
                                                       shape=[1, self.density_map_height, self.density_map_width])

        with slim.arg_scope(vgg.vgg_arg_scope()):
            r1_output = self.get_r1(regressor_input)
            r1_output = tf.scalar_mul(self.r1_out_multiplier, r1_output)
            r2_output = self.get_r2(regressor_input)
            r2_output = tf.scalar_mul(self.r1_out_multiplier, r2_output)
            r3_output = self.get_r3(regressor_input)
            r3_output = tf.scalar_mul(self.r1_out_multiplier, r3_output)

            net = tf.concat([r1_output, r2_output, r3_output], axis=3)
            with slim.arg_scope(vgg.vgg_arg_scope()):
                net = slim.conv2d(net,  1, [1, 1], scope='r123_combine')

            self.cost_regressor = tf.reduce_sum(tf.pow(
                tf.subtract(tf.squeeze(net), tf.scalar_mul(1000, tf.squeeze(regressor_output_ground_truth))), 2))
            self.sum_regressor = tf.scalar_mul(0.001, tf.reduce_sum(net))
            self.optimizer_regressor = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_regressor)

            self.saver_all = tf.train.Saver()

    def run_training(self):
        init = tf.global_variables_initializer()

        dataset = FolderDataReader(self.data_path, TrainDataLoader(self.image_height, self.image_width))
        dataset.init()

        r1_multiplers = [1, 0, 0, 1]
        r2_multiplers = [0, 1, 0, 1]
        r3_multiplers = [0, 0, 1, 1]
        multipler_index = 0

        with tf.Session() as sess:
            sess.run(init)
            if not self.from_scratch:
                self.saver_all.restore(sess, self.full_model_path)
            # ======================================== Pre-training start ==================================================
            iteration = 0
            print("========== Starting Training =========")
            while True:
                epoch_num = dataset.get_next_epoch()
                if epoch_num == multipler_index * self.PRETRAIN_EACH_EPOCHS + self.PRETRAIN_EACH_EPOCHS and multipler_index < 3:
                    multipler_index += 1
                if epoch_num == self.EPOCHS:
                    break

                datum, epoch, id = dataset.next_element()
                images = datum['images']
                density_maps = datum['density_maps']

                assert(len(images) == 9 and len(density_maps) == 9)

                for i in range(9):
                    image = images[i]
                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)
                    c1, s1, o1 = sess.run([self.cost_regressor, self.sum_regressor, self.optimizer_regressor], feed_dict={self.r1_out_multiplier: r1_multiplers[multipler_index], self.r2_out_multiplier: r2_multiplers[multipler_index], self.r3_out_multiplier: r3_multiplers[multipler_index], self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    print("\tEpoch", epoch, "Iteration", iteration, "Patch", i+1)
                    print("\tCost R1", c1, "Original sum:", sum_gt, "Predicted sum", s1)

                iteration += 1

            self.saver_all.save(sess, self.full_model_path)
            print("========== Training Complete =========")
            dataset.halt()

    def run_tests(self):
        init = tf.global_variables_initializer()

        dataset = FolderDataReader(self.test_data_path, TrainDataLoader(self.image_height, self.image_width))
        dataset.init()

        total_examples = 0
        total_absolute_error = 0
        total_square_error = 0
        total_gt_sum = 0

        gt_values = []
        output_values = []

        with tf.Session() as sess:
            sess.run(init)
            self.saver_all.restore(sess, self.full_model_path)
            iteration = 0
            while True:
                if dataset.get_next_epoch() == 1:
                    break

                datum, epoch, id = dataset.next_element()
                images = datum['images']
                complete_image = datum['complete_image']
                density_maps = datum['density_maps']

                assert(len(images) == 9 and len(density_maps) == 9)

                sum_gt_9_patches = 0
                sum_predicted_total = 0

                for i in range(9):
                    image = images[i]
                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)
                    sum_gt_9_patches += sum_gt
                    # SGD backprop through all of these0
                    sum_predicted = sess.run([self.sum_regressor], feed_dict={self.r1_out_multiplier: 1, self.r2_out_multiplier: 1, self.r3_out_multiplier: 1, self.regressor_input: [image]})
                    sum_predicted_total += sum_predicted[0]

                total_absolute_error += abs(sum_predicted_total - sum_gt_9_patches)
                total_square_error += pow(abs(sum_predicted_total - sum_gt_9_patches), 2)
                total_gt_sum += sum_gt_9_patches
                gt_values.append(sum_gt_9_patches)
                output_values.append(sum_predicted_total)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(complete_image, str(sum_gt_9_patches), (0, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(complete_image, str(sum_predicted_total), (0, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.namedWindow("Draw")
                cv2.imshow("Draw", complete_image)
                cv2.waitKey(0)

                iteration += 1
                total_examples += 1

            dataset.halt()

        mean_absolute_error = total_absolute_error / total_examples
        mean_squared_error = np.sqrt(total_square_error / total_examples)
        mean_people_per_image = total_gt_sum / total_examples

        print("MAE", mean_absolute_error)
        print("MSE", mean_squared_error)
        print("Mean people per image", mean_people_per_image)


        # XX = np.array([1, 2, 10, 100, 1000])
        # YY = np.array([1, 2, 10, 100, 1000])
        #
        # plt.plot(XX,YY)
        # plt.scatter(gt_values,output_values)
        # plt.xscale('log')
        # plt.xlabel('Ground truth')
        # plt.yscale('log')
        # plt.ylabel('Predicted output')
        # plt.show()
