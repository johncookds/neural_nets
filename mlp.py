import pandas as pd
import os
import numpy as np
import scipy as sp
import sklearn.ensemble as ske
from sklearn.cross_validation import cross_val_score
import tensorflow as tf
import inspect
from collections import OrderedDict
import time
import json

class classification_mlp(object):

	def __init__(self):
		pass

	def set_params(self, **kwargs):
		self.__dict__.update(kwargs['kwargs'])


	def set_structure(self,**kwargs):
		self.set_params(kwargs=kwargs)
		self.x = tf.placeholder("float", [None, self.n_inputs])
		self.y = tf.placeholder("float", [None, self.n_classes])
		self.weights= OrderedDict({'h1':tf.Variable(tf.random_normal([self.n_inputs,self.hidden_lvls[0]]))})
		self.biases=OrderedDict({'b1': tf.Variable(tf.random_normal([self.hidden_lvls[0]]))})
		for lvl in range(1,len(self.hidden_lvls)):
			self.biases['b'+str(lvl+1)]=tf.Variable(tf.random_normal([self.hidden_lvls[lvl]]))
			self.weights['h'+str(lvl+1)] = tf.Variable(tf.random_normal([self.hidden_lvls[lvl-1],self.hidden_lvls[lvl]]))
		self.weights['out'] = tf.Variable(tf.random_normal([self.hidden_lvls[lvl],self.n_classes]))
		self.biases['out']=tf.Variable(tf.random_normal([self.n_classes]))
        
	def build(self,relu=tf.nn.relu):
		layers={}
		current_layer=tf.add(tf.matmul(self.x,self.weights['h1']),self.biases['b1'])
		current_layer= relu(current_layer)
		for lvl in range(2,len(self.weights.keys())):
			current_layer = tf.add(tf.matmul(current_layer,self.weights['h'+str(lvl)]),self.biases['b'+str(lvl)])
			current_layer = relu(current_layer)
		return tf.add(tf.matmul(current_layer,self.weights['out']),self.biases['out'])

	def set_cost(self):
		self.pred=self.build()
		self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.pred))
 
	def save_config(self,path):
		keys=['training_epochs','batch_size','learning_rate','n_inputs','hidden_lvls','n_classes']
		json_dict={k:self.__dict__[k] for k in keys}
		self.__dict__.update({'model':path})
		if os.path.exists('/tmp/mlp_map.txt'):
			with open('/tmp/mlp_map.txt','a') as f:
				f.write(json.dumps(json_dict))
				f.write('\n')
		else:
			with open('/tmp/mlp_map.txt','w+') as f:
				f.write(json.dumps(json_dict))
				f.write('\n')

	def train(self,**kwargs):
		self.set_params(kwargs=kwargs)
		self.saver = tf.train.Saver()
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		init = tf.initialize_all_variables()
		timestamp=int(time.time())
		self.file_path='/tmp'+'/{}_mlp'.format(str(timestamp))
		self.save_config(self.file_path)
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
					_, c = sess.run([optimizer,self.cost],feed_dict={self.x:batch_x, self.y:batch_y})
					avg_cost += c / total_batch
				if epoch % self.training_epochs ==0:
					print("Epoch:", '%04d' % (epoch),"cost=", "{:.9f}".format(avg_cost))
			self.saver.save(sess,self.file_path)
		print('Model Trained')

	def test(self,file=False ,**kwargs):
		if not file:
			file=self.file_path
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
		print("Testing Acc: {}".format(str(acc)))
		return acc