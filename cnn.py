from __future__ import print_function
import pandas as pd
import os
import numpy as np
import scipy as sp
import sklearn.ensemble as ske
from sklearn.cross_validation import cross_val_score
import tensorflow as tf
import inspect
from collections import OrderedDict,defaultdict

class classification_cnn(object):

	def __init__(self):
		pass

	def set_params(self, **kwargs):
		self.__dict__.update(kwargs['kwargs'])

	def _set_convolution(self):
		self.weights=defaultdict(list)
		self.biases=defaultdict(list)
		prev_channels=1
		self.conv_out=self.n_inputs
		for lvl in range(len(self.conv_lvls)):
			side_length,channels=self.conv_lvls[lvl]
			self.weights['w'+str(lvl+1)] = tf.Variable(tf.random_normal([side_length,side_length,prev_channels,channels]))
			self.biases['b'+str(lvl+1)] = tf.Variable(tf.random_normal([channels]))
			prev_channels=channels
			self.conv_out/=self.pool_lvls[lvl]**2
		self.conv_out*=prev_channels
		self.conv_out=int(self.conv_out)
		

	def _set_fc(self):
		st=len(self.conv_lvls)
		inpt=self.conv_out
		for lvl in range(len(self.fc_lvls)):
			outpt=self.fc_lvls[lvl]
			print(inpt,outpt)
			self.weights['w'+str(st+lvl+1)]=tf.Variable(tf.random_normal([inpt,outpt]))
			self.biases['b'+str(st+lvl+1)]=tf.Variable(tf.random_normal([outpt]))
			inpt=outpt
		self.weights['out']=tf.Variable(tf.random_normal([inpt,self.n_classes]))
		self.biases['out']=tf.Variable(tf.random_normal([self.n_classes]))


	def set_structure(self,**kwargs):
		# conv_lvls(conv_lvl), [side length, channel #]
		# pool_lvls,  side length
		# fc_lvls, outputs
		self.set_params(kwargs=kwargs)
		self.x = tf.placeholder("float", [None, self.n_inputs])
		self.y = tf.placeholder("float", [None, self.n_classes])
		self.keep_prob = tf.placeholder("float")
		self._set_convolution()
		self._set_fc()

	def _conv2d(self,x,W,b,strides=1,relu=tf.nn.relu):
		x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
		x=tf.nn.bias_add(x,b)
		return relu(x)

	def _maxpool2d(self,x,k):
		return tf.nn.max_pool(x, ksize=[1,k,k,1],strides = [1,k,k,1], padding='SAME')

	def build(self,relu=tf.nn.relu):
		s_lngth=int(np.sqrt(self.n_inputs))
		conv_layer=tf.reshape(self.x, shape=[-1,s_lngth,s_lngth,1])
		for conv_lvl in range(1,len(self.conv_lvls)+1):
			conv_layer=self._conv2d(conv_layer,self.weights['w'+str(conv_lvl)],self.biases['b'+str(conv_lvl)])
			conv_layer=self._maxpool2d(conv_layer,k=self.pool_lvls[conv_lvl-1])
		fc_layer=tf.reshape(conv_layer, [-1,self.conv_out])
		for fc_lvl in range(conv_lvl+1,conv_lvl+1+len(self.fc_lvls)):
			fc_layer=tf.add(tf.matmul(fc_layer,self.weights['w'+str(fc_lvl)]),self.biases['b'+str(fc_lvl)])
		fc_layer=tf.nn.dropout(fc_layer,self.dropout)
		return tf.add(tf.matmul(fc_layer,self.weights['out']),self.biases['out'])

	def set_cost(self):
		self.pred=self.build()
		self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.pred))

	def save_config(self,path):
		keys=['training_epochs','batch_size','learning_rate','n_inputs','conv_lvls','fc_lvls','n_classes']
		json_dict={k:self.__dict__[k] for k in keys}
		self.__dict__.update({'model':path})
		if os.path.exists('/tmp/cnn_map.txt'):
			with open('/tmp/cnn_map.txt','a') as f:
				f.write(json.dumps(json_dict))
				f.write('\n')
		else:
			with open('/tmp/cnn_map.txt','w+') as f:
				f.write(json.dumps(json_dict))
				f.write('\n')

	def train(self,**kwargs):
		self.set_params(kwargs=kwargs)
		self.saver = tf.train.Saver()
		timestamp=time.time()
		self.recent_file='/tmp'+'/{}_cnn'.format(str(timestamp))
		self.save_config(self.recent_file)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		init = tf.initialize_all_variables()
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			sess.run(init)
			for epoch in range(1,self.training_epochs+1):
				avg_cost=0.
				total_batch = int(np.shape(self.data)[0]/self.batch_size)
				np.random.shuffle(self.data)
				for i in range(total_batch):
					batch_x = self.data[:,1:][i*self.batch_size:(i+1)*self.batch_size]
					lbl_y = self.data[:,0][i*self.batch_size:(i+1)*self.batch_size]
					batch_y = np.zeros((self.batch_size,self.n_classes))
					batch_y[np.arange(self.batch_size),lbl_y.astype('int')]=1
					_, c = sess.run([optimizer,self.cost],feed_dict={self.x:batch_x, self.y:batch_y,self.keep_prob:1.})
					avg_cost += c / total_batch
				if epoch % self.training_epochs ==0:
					print("Epoch:", '%04d' % (epoch),"cost=", "{:.9f}".format(avg_cost))
			self.saver.save(sess,self.recent_file)
		print('Training Finished')

	def test(self,file=False ,**kwargs):
		if not file:
			file = self.recent_file
		print('Testing...')
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			self.saver.restore(sess,file)
			correct_prediction = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			accuracy=accuracy.eval({self.x: self.test_x, self.y: self.test_y})
		return accuracy

	def run(self, **kwargs):
		self.set_params(kwargs=kwargs)
		self.set_structure()
		self.set_cost()
		self.train()
		acc=self.test()
		return acc
	
