#!python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
# download the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
MINST:
	training set:50000
	valid set: 5000
	testing set 10000
'''


class MNIST_CNN:
	def __init__(self):
		# placeholder in tensorflow
		self.keep_prob = tf.placeholder(tf.float32)
		self.x = tf.placeholder(tf.float32, shape=[None, 784])
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
		# network parameter input->32->64->1024->10(class)
		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
		self.W_conv1 = self.weight_variable([5, 5, 1, 32], 'conv1_weights')
		self.b_conv1 = self.bias_variable([32], 'conv1_bias')
		self.W_conv2 = self.weight_variable([5, 5, 32, 64], 'conv2_weights')
		self.b_conv2 = self.bias_variable([64], 'conv2_bias')
		self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], 'full1_weights')  # maxpooling two times
		self.b_fc1 = self.bias_variable([1024], 'full1_bias')
		self.W_fc2 = self.weight_variable([1024, 10], 'full2_weights')
		self.b_fc2 = self.bias_variable([10], 'full2_bias')

		# tensorlfow graph operation
		# layer 1
		self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)
		self.layer_1 = self.h_pool1
		# layer 2
		self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = self.max_pool_2x2(self.h_conv2)
		self.layer_2 = self.h_pool2
		# full connected layer
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])  # reshape to fully connected neural size, -1 means remainder size
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
		# output layer
		self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y_))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.saver = tf.train.Saver()

	def weight_variable(self, shape, name):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name)

	def bias_variable(self, shape, name):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name)

	def conv2d(self, X, W):
		return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, X):
		return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def training(self, restore=False):
		# start training
		with tf.Session() as sess:
			if restore:
				self.saver.restore(sess, "./model.ckpt")
			else:
				sess.run(tf.global_variables_initializer())
			for i in range(20000):
				batch = mnist.train.next_batch(50)
				_, train_accuracy = sess.run([self.train_step, self.accuracy], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
				print("step %d, batch's training accuracy %g" % (i, train_accuracy))
				if i % 100 == 0:
					self.aver.save(sess, "./model.ckpt")
			print("test accuracy %g" % sess.run(self.accuracy, feed_dict={
				self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))
			self.saver.save(sess, "./model.ckpt")

	def testing_data_visulization(self):
		def plot_scatter(x, labels, title, txt=False):
			plt.title(title)
			ax = plt.subplot()
			ax.scatter(x[:, 0], x[:, 1], c=labels)
			# txts = []
			if txt:
				for i in range(10):
					xtext, ytext = np.median(x[labels == i, :], axis=0)
					txt = ax.text(xtext, ytext, str(i), fontsize=24)
					txt.set_path_effects([
						PathEffects.Stroke(linewidth=5, foreground="w"),
						PathEffects.Normal()])
					# txts.append(txt)
				plt.show()

		def normalize(input):
			'''
			input should be numpy ndarray
			'''
			std = input.std()
			mean = input.mean()

			return (input - mean) / std

		def pca(input_x, k_componet, name_scope, read_eg=False, save_eg=False):
			'''
			try to implement pca by myself
			set read_eg = False to comput PCA, or it will read eigen from files
			return:
				- Z : the k priciple compnent of inptut x
			'''
			input_dim = input_x.shape
			print('input dimention {}'.format(input_dim))
			if k_componet > input_x.shape[-1]:
				print('k_componet is larger than input_x dimention')
				return

			X = normalize(input_x)
			covX = np.cov(X.T)  # [[x1, x2], [x1, x2], [x1, x2]] to [[x1, x1, x1],[x2, x2, x2]] x1, x2 are dimention
			eg_values = None
			eg_vectors = None

			if read_eg:
				try:
					eg_values = np.load('./eg_values_' + name_scope)
					eg_vectors = np.load('./eg_vectors_' + name_scope)
				except Exception:  # if file not exit, calculate...
					print('calculat eig...')
					eg_values, eg_vectors = np.linalg.eig(covX)
			else:
				print('calculat eig...')
				eg_values, eg_vectors = np.linalg.eig(covX)

			print('eg_vectors shape {}, eg_values shape {}'.format(eg_vectors.shape, eg_values.shape))
			eg_values_index = np.argsort(-eg_values)  # get the index of sort eg_values from big to small
			# print(eg_values_index[:k_componet])
			select_eg_vector = eg_vectors[:, eg_values_index[:k_componet]]
			print('select_eg_vector shape{}'.format(select_eg_vector.shape))
			Z = np.dot(X, select_eg_vector)  # numpy arrays are not matrices

			if save_eg:
				np.save('./eg_values_' + name_scope, eg_values, allow_pickle=True)
				np.save('./eg_vectors_' + name_scope, eg_vectors, allow_pickle=True)
			return Z

		def skl_PCA(x, k_componet):
			'''
				sklearn PCA: Linear dimensionality reduction using Singular Value Decomposition
			'''
			pca = PCA(n_components=k_componet)
			pca.fit(x)
			return pca.transform(x)

		layer1_reshape = tf.reshape(self.layer_1[:, :, :, :], [-1, 14 * 14 * 32])
		layer2_reshape = tf.reshape(self.layer_2[:, :, :, :], [-1, 7 * 7 * 64])
		test_size = 5000
		test_data = mnist.test.images[0:test_size, :]
		test_label = mnist.test.labels[0:test_size, :]
		test_label_index = np.argmax(test_label, axis=1)
		with tf.Session() as sess:
			self.saver.restore(sess, "./model.ckpt")
			# layer 1
			# test_layer1_pca = pca(sess.run(layer1_reshape, feed_dict={self.x: test_data}), 2, name_scope='layer1', read_eg=True, save_eg=True)
			test_layer1_pca = skl_PCA(sess.run(layer1_reshape, feed_dict={self.x: test_data}), 2)
			plot_scatter(test_layer1_pca, test_label_index, "conv layer1 with pca", txt=True)
			# layer 2
			# test_layer2_pca = pca(sess.run(layer2_reshape, feed_dict={self.x: test_data}), 2, name_scope='layer2', read_eg=True, save_eg=True)
			test_layer2_pca = skl_PCA(sess.run(layer2_reshape, feed_dict={self.x: test_data}), 2)
			plot_scatter(test_layer2_pca, test_label_index, "conv layer2 with pca", txt=True)
			# layer 3
			# dc1_pca = pca(sess.run(self.h_fc1, feed_dict={self.x: test_data}), 2, name_scope='fc_layer1', read_eg=True, save_eg=True)
			dc1_pca = skl_PCA(sess.run(self.h_fc1, feed_dict={self.x: test_data}), 2)
			plot_scatter(dc1_pca, test_label_index, "fc layer with pca", txt=True)


if __name__ == '__main__':
	cnn = MNIST_CNN()  # create instance
	cnn.training(restore=True)  # if training
	cnn.testing_data_visulization()  # if data visulization
