import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class CNN_auto:
	def __init__(self):
		# network parameter
		self.learning_rate = 0.001
		self.training_iters = 10000
		self.batch_size = 70
		self.display_step = 50
		self.dropout = 0.6
		self.weight_decay = 0.01  # for L2 norm

		# network parameter
		# 1->16->32->16->1
		self.input_layer = 1
		self.encoder_1 = 16
		self.encoder_2 = 32
		self.decoder_1 = self.encoder_1
		self.decoder_2 = self.input_layer  # output layer

		# variable
		# CNN fileter
		self.weights = {
			'conv1': self.weight_variable([5, 5, self.input_layer, self.encoder_1], 'conv1_w'),
			'conv2': self.weight_variable([5, 5, self.encoder_1, self.encoder_2], 'conv2_w'),
			'deconv1': self.weight_variable([5, 5, self.decoder_1, self.encoder_2], 'deconv1_w'),
			'deconv2': self.weight_variable([5, 5, self.decoder_2, self.decoder_1], 'deconv2_w')
		}
		self.bias = {
			'conv1': self.bias_variable([self.encoder_1], 'conv1_b'),
			'conv2': self.bias_variable([self.encoder_2], 'conv2_b'),
			'deconv1': self.bias_variable([self.decoder_1], 'deconv1_b'),
			'deconv2': self.bias_variable([self.decoder_2], 'deconv2_b'),
		}
		# placeholde
		self.keep_prob = tf.placeholder(tf.float32)
		self.x = tf.placeholder(tf.float32, shape=[None, 784])
		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

		# tensorflow operation
		_, self.reconstruct = self.build_net(self.x_image, self.weights, self.bias)
		self.cost_OP = tf.reduce_mean(tf.pow(self.reconstruct - self.x_image, 2))
		self.optimization_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_OP)
		self.saver = tf.train.Saver()

	def build_net(self, x, weights, bias):
		'''
			it will train failed if output layer use relu.
		'''
		encode_layer_1 = self.conv2d(x, weights['conv1'], bias['conv1'])
		encode_layer_1 = tf.nn.relu(encode_layer_1)
		encode_layer_1, argmax_1 = self.maxpool2d(encode_layer_1)

		encode_layer_2 = self.conv2d(encode_layer_1, weights['conv2'], bias['conv2'])
		encode_layer_2 = tf.nn.relu(encode_layer_2)
		encode_layer_2, argmax_2 = self.maxpool2d(encode_layer_2)

		enocde_layer_output = encode_layer_2
		print('enocde_layer_output shape {}'.format(enocde_layer_output.get_shape()))  # [, 7, 7, 32]
		# max unpooling
		decode_layer_1 = self.unmaxpool2d(enocde_layer_output, argmax_2)  # output [, 14, 14, 32]
		output_shape_of_dconv1 = tf.pack([tf.shape(x)[0], 14, 14, self.decoder_1])  # output channel: 16
		decode_layer_1 = self.deconv2d(decode_layer_1, weights['deconv1'], bias['deconv1'], output_shape_of_dconv1)
		decode_layer_1 = tf.nn.relu(decode_layer_1)

		decode_layer_2 = self.unmaxpool2d(decode_layer_1, argmax_1)  # output [, 28, 28, 16]
		output_shape_of_dconv2 = tf.pack([tf.shape(x)[0], 28, 28, self.decoder_2])  # output channel: 1
		decode_layer_2 = self.deconv2d(decode_layer_2, weights['deconv2'], bias['deconv2'], output_shape_of_dconv2)
		decode_layer_2 = tf.nn.tanh(decode_layer_2)

		decode_layer_output = decode_layer_2
		return enocde_layer_output, decode_layer_output

	def weight_variable(self, shape, name):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name)

	def bias_variable(self, shape, name):
		initial = tf.random_normal(shape)
		return tf.Variable(initial, name)

	def conv2d(self, x, w, b, strides=1):
		x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return x

	def deconv2d(self, x, w, b, output_shape, strides=1):
		x = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return x

	def maxpool2d(self, x, k=2):
		_, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, k, k, 1], strides=[
			1, k, k, 1], padding='SAME')
		x = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[
			1, k, k, 1], padding='SAME')
		return x, argmax

	def unmaxpool2d(self, updates, argmax, ksize=[1, 2, 2, 1]):
		'''
			argmax indeics:
				[b, y, x, f] -> ((b * height + y) * width + x) * channels + f
		'''
		argmax = tf.to_int32(argmax)
		input_shape = tf.shape(updates, out_type=tf.int32)
		# calculate output shape
		output_shape = (
			input_shape[0],
			input_shape[1] * ksize[1],
			input_shape[2] * ksize[2],
			input_shape[3])

		one_like_mask = tf.ones_like(argmax, dtype=tf.int32)
		batch_shape = tf.pack([input_shape[0], 1, 1, 1])
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
		b = one_like_mask * batch_range
		y = argmax // (output_shape[2] * output_shape[3])
		x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]  # % and // at the sa,e prcedence
		feature_range = tf.range(output_shape[3], dtype=tf.int32)
		f = one_like_mask * feature_range
		updates_size = tf.size(updates)
		indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
		print('indices shape {}'.format(indices.get_shape()))
		values = tf.reshape(updates, [updates_size])  # faltturn
		print('value shape {}'.format(values.get_shape()))
		ret = tf.scatter_nd(indices, values, output_shape)
		return ret

	def max_unpool_2x2(x, output_shape):
		out = tf.concat_v2([x, tf.zeros_like(x)], 3)
		out = tf.concat_v2([out, tf.zeros_like(out)], 2)
		out_size = output_shape
		return tf.reshape(out, out_size)

	def plot_decoder_ouput(self):
		trainimg = mnist.train.images
		trainlabel = mnist.train.labels
		trainimg = trainimg[:10]
		trainlabel = trainlabel[:10]
		print('trainlabel shape{}'.format(trainlabel.shape))
		plot_size = 4
		with tf.Session() as sess:
			try:
				self.saver.restore(sess, "./model.ckpt")
			except Exception:
				print("can't find model in current dir")
				return
			origin_img = np.reshape(trainimg, (-1, 28, 28))  # 28*28 pixel
			origin_label = np.argmax(trainlabel, axis=1)
			print('origin_label shape{}'.format(origin_label.shape))
			decode_img = sess.run(self.reconstruct, feed_dict={self.x: trainimg})
			decode_img = np.reshape(decode_img, (-1, 28, 28))  # 28*28 pixel
			for i in range(plot_size):
				plt.matshow(origin_img[i], cmap=plt.get_cmap('gray'))  # plot matrix as image
				plt.matshow(decode_img[i], cmap=plt.get_cmap('gray'))  # cmp is color map
				plt.show()

	def training(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(self.training_iters):
				batch = mnist.train.next_batch(self.batch_size)
				_, loss = sess.run([self.optimization_op, self.cost_OP], feed_dict={self.x: batch[0]})
				if epoch % 100 == 0:
					print('epoch {} loss {}'.format(epoch, loss))
					self.saver.save(sess, './model.ckpt')
			print('optimization finish!')
			test_loss = sess.run(self.cost_OP, feed_dict={self.x: mnist.test.images})
			print('final loss: {}'.format(test_loss))
			self.saver.save(sess, './model.ckpt')


if __name__ == '__main__':
	cnauto = CNN_auto()
	cnauto.training()
	cnauto.plot_decoder_ouput()
