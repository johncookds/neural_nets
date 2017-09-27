import pandas as pd
import os
import numpy as np
import scipy as sp
import sklearn.ensemble as ske
from sklearn.cross_validation import cross_val_score
import tensorflow as tf
from __future__ import print_function
import inspect
from collections import OrderedDict

class classification_mlp(object):

	def __init__(self):
		pass


	def set_structure(self,n_inputs,hidden_lvls,n_classes):
		self.n_classes=n_classes
		self.x = tf.placeholder("float", [None, n_inputs])
		self.y = tf.placeholder("float", [None, n_classes])
		self.weights= OrderedDict({'h1':tf.Variable(tf.random_normal([n_inputs,hidden_lvls[0]]))})
		self.biases=OrderedDict({'b1': tf.Variable(tf.random_normal([hidden_lvls[0]]))})
		for lvl in range(1,len(hidden_lvls)):
			self.biases['b'+str(lvl+1)]=tf.Variable(tf.random_normal([hidden_lvls[lvl]]))
			self.weights['h'+str(lvl+1)] = tf.Variable(tf.random_normal([hidden_lvls[lvl-1],hidden_lvls[lvl]]))
		self.weights['out'] = tf.Variable(tf.random_normal([hidden_lvls[lvl],n_classes]))
		self.biases['out']=tf.Variable(tf.random_normal([n_classes]))
		print(self.weights)
		print(self.biases)
        
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

	def load_data(self,data):
		self.data=data

        
	def train(self,training_epochs=1,batch_size=100,learning_rate=.001):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
		init = tf.initialize_all_variables()
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			sess.run(init)
			for epoch in range(training_epochs):
				avg_cost=0.
				total_batch = int(np.shape(self.data)[0]/batch_size)
				np.random.shuffle(self.data)
				for i in range(total_batch):
					batch_x = self.data[:,1:][i*batch_size:(i+1)*batch_size]
					lbl_y = self.data[:,0][i*batch_size:(i+1)*batch_size]
					batch_y = np.zeros((batch_size,self.n_classes))
					batch_y[np.arange(batch_size),lbl_y.astype('int')]=1
					_, c = sess.run([optimizer,self.cost],feed_dict={self.x:batch_x, self.y:batch_y})
					avg_cost += c / total_batch
				if epoch % 1 ==0:
					print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
			print("Optimization finished")
			correct_prediction = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		return accuracy