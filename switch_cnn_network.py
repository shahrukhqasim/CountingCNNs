import vgg
import tensorflow as tf
from silknet import *
from silknet import LoadInterface
from interface import implements
from silknet import FolderDataReader
import cv2

slim = tf.contrib.slim


class TrainDataSwitchValuesWriter(implements(WriteInterface)):
    def write_datum(self, full_path, object):
        indices_x = object['indices_x']
        indices_y = object['indices_y']
        switch_values = object['switch_values']
        assert(len(indices_x) == 9 and len(indices_y) == 9 and len(switch_values) == 9)
        for i in range(9):
            with open(os.path.join(full_path,str(indices_x[i])+'_'+str(indices_y[i])+'.swt'), "w") as text_file:
                text_file.write(switch_values[i])


class TrainDataLoader(implements(LoadInterface)):
    def load_map(self, path):
        map = np.fromfile(path, dtype=np.float32)
        total_len = np.size(map)
        one_side_length = np.sqrt(total_len)
        assert(np.floor(one_side_length) == int(one_side_length))
        one_side_length = int(one_side_length)
        return np.reshape(map, (one_side_length, one_side_length))

    def load_datum(self, full_path):
        images = []
        density_maps = []
        indices_x = []
        indices_y = []

        for i in range(3):
            for j in range(3):
                images.append(cv2.resize(cv2.imread(os.path.join(full_path, '%d_%d.jpg' % (i, j))), (200,200)))
                density_maps.append(self.load_map(os.path.join(full_path, '%d_%d.bin' % (i, j))))
                indices_x.append(i)
                indices_y.append(j)

        datum = dict()
        datum['images'] = images
        datum['density_maps'] = density_maps
        datum['indices_x'] = indices_x
        datum['indices_y'] = indices_y

        return datum



class TrainDataLoaderWithSwitch(implements(LoadInterface)):
    def load_map(self, path):
        map = np.fromfile(path, dtype=np.float32)
        total_len = np.size(map)
        one_side_length = np.sqrt(total_len)
        assert(np.floor(one_side_length) == int(one_side_length))
        one_side_length = int(one_side_length)
        return np.reshape(map, (one_side_length, one_side_length))

    def load_datum(self, full_path):
        images = []
        density_maps = []
        indices_x = []
        indices_y = []
        switch_values = []

        for i in range(3):
            for j in range(3):
                with open(os.path.join(full_path, '%d_%d.swt' % (i, j)), 'r') as content_file:
                    content = content_file.read()
                images.append(cv2.resize(cv2.imread(os.path.join(full_path, '%d_%d.jpg' % (i, j))), (200,200)))
                density_maps.append(self.load_map(os.path.join(full_path, '%d_%d.bin' % (i, j))))
                switch_values.append(int(content))
                indices_x.append(i)
                indices_y.append(j)

        datum = dict()
        datum['images'] = images
        datum['density_maps'] = density_maps
        datum['indices_x'] = indices_x
        datum['indices_y'] = indices_y

        return datum


class SwitchCnnNetwork:
    def __init__(self):
        self.image_width = 200
        self.image_height = 200
        self.density_map_width = int(self.image_width / 4)
        self.density_map_height = int(self.image_height / 4)
        self.learning_rate = 0.00001
        self.saver_vgg = None
        self.saver_all = None
        # TODO: Parameter
        self.vgg_path = '/home/srq/Projects/FishRender/scripts/train/logs/vgg_16.ckpt'
        self.full_model_path = 'models/model_full.ckpt'
        self.training_mode = 0  # 0 for pre-training, 1 for differential training and 2 for coupled training
        self.from_scratch = False
        self.data_path = '/home/srq/Datasets/people-counting-1/train-two'
        self.PRETRAIN_EPOCHS = 10
        self.DIFFERENTIAL_TRAIN_EPOCHS = 10
        self.COUPLED_TRAIN_EPOCHS = 10

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

    def get_r2(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 20, [7, 7], scope='r2_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r2_p1')
            net = slim.conv2d(net, 40, [5, 5], scope='r2_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r2_p2')
            net = slim.conv2d(net, 20, [5, 5], scope='r2_c3')
            net = slim.conv2d(net, 10, [5, 5], scope='r2_c4')
            net = slim.conv2d(net, 1, [1, 1], scope='r2_c5')
        return net

    def get_r3(self, x):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.conv2d(x, 24, [5, 5], scope='r3_c1')
            net = slim.max_pool2d(net, [2, 2], scope='r3_p1')
            net = slim.conv2d(net, 48, [3, 3], scope='r3_c2')
            net = slim.max_pool2d(net, [2, 2], scope='r3_p2')
            net = slim.conv2d(net, 24, [3, 3], scope='r3_c3')
            net = slim.conv2d(net, 12, [3, 3], scope='r3_c4')
            net = slim.conv2d(net, 1, [1, 1], scope='r3_c5')
        return net

    def construct_graphs(self):
        classifier_input = self.classifier_input = tf.placeholder("float32", shape=[1, 224, 224, 3])
        classifier_output_gt = self.classifier_output_gt = tf.placeholder("float32", shape=[1, 3])
        regressor_input = self.regressor_input = tf.placeholder("float32", shape=[1, self.image_height, self.image_width, 3])
        regressor_output_ground_truth = self.regressor_output_ground_truth = tf.placeholder("float32",
                                                       shape=[1, self.density_map_height, self.density_map_width])

        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = vgg.vgg_16(classifier_input)
            self.saver_vgg = tf.train.Saver()
            classifier_logits = slim.conv2d(net, 3, [1, 1], activation_fn=None, normalizer_fn=None)
            classifier_logits = tf.squeeze(classifier_logits, [1, 2])
            r1_output = self.get_r1(regressor_input)
            r2_output = self.get_r2(regressor_input)
            r3_output = self.get_r3(regressor_input)

            self.y_classifier_output_maxed = tf.arg_max(classifier_logits, 1)

            self.cost_classifier = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=classifier_output_gt, logits=classifier_logits))
            self.optimizer_classifier = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.cost_classifier)

            self.cost_r1 = tf.reduce_sum(tf.pow(
                tf.subtract(tf.squeeze(r1_output), tf.scalar_mul(1000, tf.squeeze(regressor_output_ground_truth))), 2))
            self.sum_r1 = tf.scalar_mul(0.001, tf.reduce_sum(r1_output))
            self.optimizer_r1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_r1)

            self.cost_r2 = tf.reduce_sum(tf.pow(
                tf.subtract(tf.squeeze(r2_output), tf.scalar_mul(1000, tf.squeeze(regressor_output_ground_truth))), 2))
            self.sum_r2 = tf.scalar_mul(0.001, tf.reduce_sum(r2_output))
            self.optimizer_r2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_r2)

            self.cost_r3 = tf.reduce_sum(tf.pow(
                tf.subtract(tf.squeeze(r3_output), tf.scalar_mul(1000, tf.squeeze(regressor_output_ground_truth))), 2))
            self.sum_r3 = tf.scalar_mul(0.001, tf.reduce_sum(r3_output))
            self.optimizer_r3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_r3)

            self.saver_all = tf.train.Saver()

    def run_training(self):
        init = tf.global_variables_initializer()

        dataset = FolderDataReader(self.data_path, TrainDataLoader())
        dataset.init()

        with tf.Session() as sess:
            sess.run(init)
            if self.from_scratch:
                self.saver_vgg.restore(sess, self.vgg_path)
            else:
                self.saver_all.restore(sess, self.full_model_path)
            # ======================================== Pre-training start ==================================================
            iteration = 0
            print("========== Starting Pre-Training =========")
            while True:
                if dataset.get_next_epoch() == self.PRETRAIN_EPOCHS:
                    break

                datum, epoch, id = dataset.next_element()
                images = datum['images']
                density_maps = datum['density_maps']

                assert(len(images) == 9 and len(density_maps) == 9)

                for i in range(9):
                    image = images[i]
                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)
                    # SGD backprop through all of these0
                    c1, s1, o1 = sess.run([self.cost_r1, self.sum_r1, self.optimizer_r1], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    c2, s2, o2 = sess.run([self.cost_r2, self.sum_r2, self.optimizer_r2], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    c3, s3, o3 = sess.run([self.cost_r3, self.sum_r3, self.optimizer_r3], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    print("\tEpoch", epoch, "Iteration", iteration, "Patch", i+1)
                    print("\tCost R1", c1, "Original sum:", sum_gt, "Predicted sum", s1)
                    print("\tCost R2", c2, "Original sum:", sum_gt, "Predicted sum", s2)
                    print("\tCost R3", c3, "Original sum:", sum_gt, "Predicted sum", s3)

                iteration += 1

            self.saver_all.save(sess, self.full_model_path)
            # print("========== Pre-Training Complete =========")
            dataset.halt()

            dataset = FolderDataReader(self.data_path, TrainDataLoader())
            dataset.init()

            # ======================================== Differential training start =========================================
            iteration = 0
            print("========== Starting Differential Training =========")
            while True:
                if dataset.get_next_epoch() == self.PRETRAIN_EPOCHS:
                    break

                datum, epoch, id = dataset.next_element()
                images = datum['images']
                density_maps = datum['density_maps']

                assert(len(images) == 9 and len(density_maps) == 9)

                for i in range(9):
                    image = images[i]
                    density_map = density_maps[i]
                    sum_gt = np.sum(density_map)

                    s1 = sess.run([self.sum_r1], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    s2 = sess.run([self.sum_r2], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                    s3 = sess.run([self.sum_r3], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})

                    print("\tSum of regressor 1", s1)
                    print("\tSum of regressor 2", s2)
                    print("\tSum of regressor 3", s3)

                    d1 = abs(s1[0] - sum_gt)
                    d2 = abs(s2[0] - sum_gt)
                    d3 = abs(s3[0] - sum_gt)

                    switch_value = np.argmax(np.array([d1, d2, d3]))

                    print("\tEpoch", epoch, "Iteration", iteration)

                    if switch_value == 0:
                        c1, s1, o1 = sess.run([self.cost_r1, self.sum_r1, self.optimizer_r1], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                        print("\tCost R1", c1, "Original sum:", sum_gt, "Predicted sum", s1)

                    elif switch_value == 1:
                        c2, s2, o2 = sess.run([self.cost_r2, self.sum_r2, self.optimizer_r2], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                        print("\tCost R2", c2, "Original sum:", sum_gt, "Predicted sum", s2)
                    else:
                        c3, s3, o3 = sess.run([self.cost_r3, self.sum_r3, self.optimizer_r3], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                        print("\tCost R3", c3, "Original sum:", sum_gt, "Predicted sum", s3)

                print()

                iteration += 1

            self.saver_all.save(sess, self.full_model_path)
            print("========== Differential Training Complete =========")
            # ========================================= Differential training end ==========================================

            dataset.halt()
            dataset = FolderDataReader(self.data_path, TrainDataLoader())
            dataset.init()

            switch_values_writer = FolderDataWriter(self.data_path, TrainDataSwitchValuesWriter())

            # ========================================= Coupled training start =============================================
            print("========== Starting Coupled Training =========")
            # saver_vgg.restore(sess, vgg_path)
            for epochs in range(self.COUPLED_TRAINING_EPOCHS):
                # Generate labels
                iteration = 0
                print("\t Generating GT")
                epochs_elapsed_old = dataset.get_next_epoch()
                while True:
                    if epochs_elapsed_old.get_next_epoch() == epochs_elapsed_old + 1:
                        break
                    datum, epoch, id = dataset.next_element()
                    images = datum['images']
                    density_maps = datum['density_maps']
                    switch_values = []

                    datum['switch_values'] = switch_values

                    assert (len(images) == 9 and len(density_maps) == 9)

                    for i in range(9):
                        image = images[i]
                        density_map = density_maps[i]
                        sum_gt = np.sum(density_map)

                        s1 = sess.run([self.sum_r1], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                        s2 = sess.run([self.sum_r2], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})
                        s3 = sess.run([self.sum_r3], feed_dict={self.regressor_input: [image], self.regressor_output_ground_truth: [density_map]})

                        print("\tSum of GT", sum_gt)
                        print("\tSum of regressor 1", s1)
                        print("\tSum of regressor 2", s2)
                        print("\tSum of regressor 3", s3)

                        d1 = abs(s1[0] - sum_gt)
                        d2 = abs(s2[0] - sum_gt)
                        d3 = abs(s3[0] - sum_gt)

                        switch_value = np.argmax(np.array([d1, d2, d3]))
                        print("\tChose switch value", switch_value)

                        switch_values.append(density_map)

                    switch_values_writer.write_datum(id, datum)

                dataset.halt()

                dataset = FolderDataReader(self.data_path, TrainDataLoaderWithSwitch())
                dataset.init()

                iteration = 0
                epochs_elapsed_old = dataset.get_next_epoch()
                print("\t Training switch")
                while True:
                    if data_with_switch_values.get_next_epoch() == epochs_elapsed_old + 1:
                        break

                    x, y, epoch, swt_one_hot = data_with_switch_values.next_sample(1)

                    x_small = [np.resize(x[0], (224, 224, 3))]

                    c, o, logits = sess.run([cost_classifier, optimzer_classifier, y_classifier_output],
                                            feed_dict={x_classifier_input: x_small,
                                                       y_classifier_output_gt: swt_one_hot})

                    print("\tEpoch", epoch[0], "Iteration", iteration)
                    print("\tCost classifier", c)
                    print("\tLogits", logits)

                    iteration += 1

                iteration = 0
                epochs_elapsed_old = data_with_switch_values.get_next_epoch()
                print("\t Switched differential training")
                while True:
                    if data_with_switch_values.get_next_epoch() == epochs_elapsed_old + 1:
                        break

                    x, y, epoch, swt_one_hot = data_with_switch_values.next_sample(1)
                    sum_gt = np.sum(y[0])

                    x_small = [np.resize(x[0], (224, 224, 3))]

                    switch_value = sess.run([y_classifier_output_maxed],
                                            feed_dict={x_classifier_input: x_small})

                    print("\tEpoch", epoch[0], "Iteration", iteration)
                    if switch_value[0] == 0:
                        c1, s1, o1 = sess.run([cost_r1, sum_r1, optimizer_r1],
                                              feed_dict={x_regressor_input: x, y_regressor_output_ground_truth: y})
                        print("\tCost R1", c1, "Original sum:", sum_gt, "Predicted sum", s1)
                    elif switch_value[0] == 1:
                        c2, s2, o2 = sess.run([cost_r2, sum_r2, optimizer_r2],
                                              feed_dict={x_regressor_input: x, y_regressor_output_ground_truth: y})
                        print("\tCost R2", c2, "Original sum:", sum_gt, "Predicted sum", s2)
                    elif switch_value[0] == 2:
                        c3, s3, o3 = sess.run([cost_r3, sum_r3, optimizer_r3],
                                              feed_dict={x_regressor_input: x, y_regressor_output_ground_truth: y})
                        print("\tCost R3", c3, "Original sum:", sum_gt, "Predicted sum", s3)
                    else:
                        0 / 0

                    print()

                    iteration += 1

    def run_tests(self):
        pass
