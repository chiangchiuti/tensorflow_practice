import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class auto_encoder:
	def __init__(self, input_X_shape):
		self.x_width, self.x_height, self.x_channel = input_X_shape   # unpacking list

		# tensorflow placeholder
		self.x = tf.placeholder(tf.float32, shape=[None, self.x_width * self.x_height * self.x_channel])

		# network parameter
		self.learning_rate = 0.001
		self.iteration = 10000
		self.input_layer = self.x_width * self.x_height * self.x_channel
		self.encoder_1 = 300
		self.encoder_2 = 100
		self.encoder_3 = 2
		self.decoder_1 = 100
		self.decoder_2 = 300
		self.decoder_3 = self.input_layer  # the same as input_layer
		self.weights = {
			'encoder_1': self.weight_variable([self.input_layer, self.encoder_1], 'encoder_1_w'),
			'encoder_2': self.weight_variable([self.encoder_1, self.encoder_2], 'encoder_2_w'),
			'encoder_3': self.weight_variable([self.encoder_2, self.encoder_3], 'encoder_3_w'),
			'decoder_1': self.weight_variable([self.encoder_3, self.decoder_1], 'dncoder_1_w'),
			'decoder_2': self.weight_variable([self.decoder_1, self.decoder_2], 'dncoder_3_w'),
			'decoder_3': self.weight_variable([self.decoder_2, self.decoder_3], 'dncoder_3_w'),
		}
		self.bias = {
			'encoder_1': self.bias_variable([self.encoder_1], 'encoder_1_b'),
			'encoder_2': self.bias_variable([self.encoder_2], 'encoder_2_b'),
			'encoder_3': self.bias_variable([self.encoder_3], 'encoder_3_b'),
			'decoder_1': self.bias_variable([self.decoder_1], 'dncoder_1_b'),
			'decoder_2': self.bias_variable([self.decoder_2], 'dncoder_3_b'),
			'decoder_3': self.bias_variable([self.decoder_3], 'dncoder_3_b'),
		}

		# tensorflow operator
		self.elayer_1 = tf.nn.relu(tf.matmul(self.x, self.weights['encoder_1']) + self.bias['encoder_1'])
		self.elayer_2 = tf.nn.relu(tf.matmul(self.elayer_1, self.weights['encoder_2']) + self.bias['encoder_2'])
		self.elayer_3 = tf.nn.relu(tf.matmul(self.elayer_2, self.weights['encoder_3']) + self.bias['encoder_3'])
		self.endcoder_layer_output = self.elayer_3
		self.delayer_1 = tf.nn.relu(tf.matmul(self.endcoder_layer_output, self.weights['decoder_1']) + self.bias['decoder_1'])
		self.delayer_2 = tf.nn.relu(tf.matmul(self.delayer_1, self.weights['decoder_2']) + self.bias['decoder_2'])
		self.delayer_3 = tf.nn.relu(tf.matmul(self.delayer_2, self.weights['decoder_3']) + self.bias['decoder_3'])
		self.decoder_layer_outtput = self.delayer_3

		self.loss_OP = tf.reduce_mean(tf.pow(self.decoder_layer_outtput - self.x, 2))  # MSE
		self.optimizer_OP = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_OP)
		self.init_OP = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

		self.encoder_classify = tf.nn.softmax(self.endcoder_layer_output)

	def weight_variable(self, shape, name):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name)

	def bias_variable(self, shape, name):
		initial = tf.random_normal(shape)
		return tf.Variable(initial, name)

	def plot_decoder_ouput(self):
		'''
		only can plot two dimention's x
		'''
		def plot_scatter(x, labels, title, txt=False):
			def normalize(input):
				'''
				input should be numpy ndarray
				'''
				std = input.std()
				mean = input.mean()
				return (input - mean) / std
			if x.shape[-1] != 2:
				print('the dimention should be two')
				return
			x = normalize(x)
			plt.title(title)
			ax = plt.subplot()
			ax.scatter(x[:, 0], x[:, 1], c=labels)
			# txts = []
			if txt:
				for i in range(10):
					xtext, ytext = np.median(x[labels == i, :], axis=0)  # should be two dimention
					txt = ax.text(xtext, ytext, str(i), fontsize=24)
					txt.set_path_effects([
						PathEffects.Stroke(linewidth=5, foreground="w"),
						PathEffects.Normal()])
					# txts.append(txt)
				plt.show()

		trainimg = mnist.train.images
		trainlabel = mnist.train.labels
		print('trainlabel shape{}'.format(trainlabel.shape))
		plot_size = 4
		with tf.Session() as sess:
			self.saver.restore(sess, "./model.ckpt")
			origin_img = np.reshape(trainimg, (-1, 28, 28))  # 28*28 pixel
			origin_label = np.argmax(trainlabel, axis=1)
			print('origin_label shape{}'.format(origin_label.shape))
			decode_img = sess.run(self.decoder_layer_outtput, feed_dict={self.x: trainimg})
			decode_img = np.reshape(decode_img, (-1, 28, 28))  # 28*28 pixel
			for i in range(plot_size):
				plt.matshow(origin_img[i], cmap=plt.get_cmap('gray'))  # plot matrix as image
				plt.matshow(decode_img[i], cmap=plt.get_cmap('gray'))  # cmp is color map
				plt.show()

			# plot  encode layer ouput 's scatter
			encode_img = sess.run(self.endcoder_layer_output, feed_dict={self.x: trainimg})
			print('encode_img shape{}'.format(encode_img.shape))
			plot_scatter(encode_img, origin_label, 'encode_layer', txt=True)

	# tensorflow network
	def auto_encoder(self):
		with tf.Session() as sess:
			sess.run(self.init_OP)
			for epoch in range(self.iteration):
				batch = mnist.train.next_batch(50)
				_, loss = sess.run([self.optimizer_OP, self.loss_OP], feed_dict={self.x: batch[0]})

				if epoch % 100 == 0:
					print('step {} loss {}'.format(epoch, loss))
					self.saver.save(sess, './model.ckpt')
			print('optimization finish!')
			test_loss = sess.run(self.loss_OP, feed_dict={self.x: mnist.test.images})
			print('final loss {}'.format(test_loss))


if __name__ == '__main__':
	auto = auto_encoder([28, 28, 1])
	auto.auto_encoder()  # training
	auto.plot_decoder_ouput()  # plot training reuslt by restor model

