import os
import time
import tensorflow as tf
import csv
from module import *

class network(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.input_c_dim = args.input_nc
        self.data_list_dir_train = args.data_list_dir_train
        self.data_list_dir_test = args.data_list_dir_test
        self.phase = args.phase
        self.dataset_name = args.dataset_name
        self._build_model()
        if self.phase == 'train':
            self.saver = tf.train.Saver(max_to_keep=2)


    def _build_model(self): 
        if self.phase == 'train':
            self.batch_train_images, self.batch_train_labels, self.num_train_images = self.build_input_batch_op(self.data_list_dir_train, batch_size = self.batch_size)
            # make the model
            self.model_train = convNet(self.batch_train_images, 10, 0.25, False, True)
        
            # create the loss function
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.model_train, labels=tf.cast(self.batch_train_labels, dtype=tf.int32)))

        self.batch_test_images, self.batch_test_labels, self.num_test_images = self.build_input_batch_op(self.data_list_dir_test, batch_size = self.batch_size)
        self.model_test = convNet(self.batch_test_images, 10, 0.25, True, False)
        
        #accuracy
        correct_prediction = tf.equal(self.batch_test_labels, tf.cast(tf.argmax(self.model_test, 1), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    @staticmethod
    def read_labeled_image_list(image_list_file):
        """
        Reads a .csv file containin paths and labels
        """
        filenames = []
        labels = []
        count = 0
        with open(image_list_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                count += 1
                filenames.append(row[0])
                labels.append(int(row[1]))

        return filenames, labels, count

    @staticmethod
    def read_images_from_disk(input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_png(file_contents, channels=3)

        return example, label

    def build_input_batch_op(self, image_list_file, batch_size, num_epochs=None):
        files_list, labels_list, num_images = network.read_labeled_image_list(image_list_file)

        images = tf.convert_to_tensor(files_list, dtype=tf.string)
        labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)

        image, label = network.read_images_from_disk(input_queue)

        image.set_shape([self.image_size, self.image_size, self.input_c_dim])
        image = tf.cast(image, tf.float32)

        # Can do preprocessing here
        image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size)

        return image_batch, label_batch, num_images

    def train(self, args):
        """Train netowrk"""

        #optimiser
        self.optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.loss)

        #summaries
        tf.summary.image("Input", self.batch_train_images)
        tf.summary.scalar('loss', self.loss)

        
        init_op = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter(args.checkpoint_dir)
        summary_op = tf.summary.merge_all()

        counter = 0
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print("Succesfully loaded prior checkpoint")
        else:
            print("FAILED to load prior checkpoint")

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        for epoch in range(args.epoch):
            print('Start epoch: {}'.format(epoch))
            batch_idxs = self.num_train_images / self.batch_size

            for idx in range(0, batch_idxs):

                # run update
                loss, _ = self.sess.run([tf.reduce_sum(self.loss), self.optim])
                
                
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %2.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time, loss)))
                
                if counter % 10 == 1:
                    summary_string = self.sess.run(summary_op)#,feed_dict={self.fake_A_sample:fake_A,self.fake_B_sample:fake_B})
                    self.writer.add_summary(summary_string,counter)

            #run test
            batch_idxs = self.num_test_images / self.batch_size
            acc = 0
            for idx in range(0, batch_idxs):
                acc += self.accuracy.eval()
            print(("Accuracy: %1.5f") % (acc / idx))

            #save the model               
            self.save(args.checkpoint_dir, counter)
          
        coord.request_stop()
        coord.join(stop_grace_period_secs=10)

    def save(self, checkpoint_dir, step):
        model_name = "%s_%s" % (self.dataset_name, self.image_size)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        def get_var_to_restore_list(ckpt_path, mask=[], prefix=""):
            """
            Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
            Args:
                ckpt_path: path to the ckpt model to be restored
                mask: list of layers to skip
                prefix: prefix string before the actual layer name in the graph definition
            """
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_dict = {}
            for v in variables:
                name = v.name[:-2]
                skip=False
                #check for skip
                for m in mask:
                    if m in name:
                        skip=True
                        continue
                if not skip:
                    variables_dict[v.name[:-2]] = v
            #print(variables_dict)
            reader = tf.train.NewCheckpointReader(ckpt_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            var_to_restore = {}
            for key in var_to_shape_map:
                #print(key)
                if prefix+key in variables_dict.keys():
                    var_to_restore[key] = variables_dict[prefix+key]
            return var_to_restore

        print("Reading checkpoint")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            savvy = tf.train.Saver(var_list=get_var_to_restore_list(ckpt.model_checkpoint_path))
            savvy.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self, args):
        print 'not implemented'