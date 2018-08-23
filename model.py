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
        self._build_model()

    def _build_model(self, phase='train'): 
        if phase == 'train':
            self.batch_train_images, self.batch_train_labels = self.build_input_batch_op(self.data_list_dir_train, batch_size = self.batch_size)
            # make the model
            self.model_train = convNet(self.batch_train_images, 10, 0.25, False, True)
        
            # create the loss function
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.model_train, labels=tf.cast(self.batch_train_labels, dtype=tf.int32)))

        self.batch_test_images, self.batch_test_labels = self.build_input_batch_op(self.data_list_dir_test, batch_size = self.batch_size)
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
        with open(image_list_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                filenames.append(row[0])
                
                labels.append(int(row[1]))

        return filenames, labels

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
        files_list, labels_list = network.read_labeled_image_list(image_list_file)

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

        return image_batch, label_batch

    def train(self, args):
        """Train netowrk"""

        #optimiser
        self.optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.loss)

        
        init_op = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.sess.run(init_op)

        counter = 0
        start_time = time.time()

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        for epoch in range(args.epoch):
            print('Start epoch: {}'.format(epoch))
            batch_idxs = args.num_sample

            for idx in range(0, batch_idxs):

                # run update
                loss, _ = self.sess.run([tf.reduce_sum(self.loss), self.optim])#,feed_dict={self.input_images : self.batch_images, \
                                           #            self.input_labels : self.batch_labels,})
                
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %2.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time, loss)))

            #run test
            [test_loss] = self.sess.run([self.accuracy])
            print(("Test loss: %1.5f") % test_loss)
          


        coord.request_stop()
        coord.join(stop_grace_period_secs=10)

    # def test(self, args):
    #     """Test cyclegan""" 
    #     sample_op, sample_path,im_shape = self.build_input_image_op(self.dataset_dir,is_test=True,num_epochs=1)
    #     sample_batch,path_batch,im_shapes = tf.train.batch([sample_op,sample_path,im_shape],batch_size=self.batch_size,num_threads=4,capacity=self.batch_size*50,allow_smaller_final_batch=True)
    #     gen_name='generatorA2B' if args.which_direction=="AtoB" else 'generatorB2A'
    #     cycle_image_batch = self.generator(sample_batch,self.options,name=gen_name)

    #     #init everything
    #     self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

    #     #start queue runners
    #     coord = tf.train.Coordinator()
    #     tf.train.start_queue_runners()
    #     print('Thread running')

    #     if self.load(args.checkpoint_dir):
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")

    #     # write html for visual comparison
    #     if not os.path.exists(args.test_dir): #python 2 is dumb...
    #         os.makedirs(args.test_dir)

    #     index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
    #     index = open(index_path, "w+")
    #     index.write("<html><body><table><tr>")
    #     index.write("<th>name</th><th>input</th><th>output</th></tr>")

    #     print('Starting')
    #     batch_num=0
    #     while True:
    #         try:
    #             print('Processed images: {}'.format(batch_num*args.batch_size), end='\r')
    #             fake_imgs,sample_images,sample_paths,im_sps = self.sess.run([cycle_image_batch,sample_batch,path_batch,im_shapes])
    #             #iterate over each sample in the batch
    #             for rr in range(fake_imgs.shape[0]):
    #                 #create output destination
    #                 dest_path = sample_paths[rr].decode('UTF-8').replace(self.dataset_dir,args.test_dir)
    #                 parent_destination = os.path.abspath(os.path.join(dest_path, os.pardir))
    #                 if not os.path.exists(parent_destination):
    #                     os.makedirs(parent_destination)

    #                 fake_img = ((fake_imgs[rr]+1)/2)*255
    #                 im_sp = im_sps[rr]
    #                 fake_img = misc.imresize(fake_img,(im_sp[0],im_sp[1]))
    #                 misc.imsave(dest_path,fake_img)
    #                 index.write("<td>%s</td>" % os.path.basename(sample_paths[rr].decode('UTF-8')))
    #                 index.write("<td><img src='%s'></td>" % (sample_paths[rr].decode('UTF-8')))
    #                 index.write("<td><img src='%s'></td>" % (dest_path))
    #                 index.write("</tr>")
    #             batch_num+=1
    #         except Exception as e:
    #             print(e)
    #             break;

    #     print('Elaboration complete')
    #     index.close()
    #     coord.request_stop()
    #     coord.join(stop_grace_period_secs=10)