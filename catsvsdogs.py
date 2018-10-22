#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author- Yang Qiao
import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm

TRAIN_DIR='/Users/joe/Desktop/BU_2018_fall/EC601/miniProject2/train'
TEST_DIR='/Users/joe/Desktop/BU_2018_fall/EC601/miniProject2/test'
IMG_SIZE=50
LR=0.0001

MODEL_NAME='dogscats-{}-{}.model'.format(LR,'2conv-basic')

def label_img(img):
	word_label=img.split('.')[0]
	if word_label== 'cat':return [1,0]
	elif word_label== 'dog':return [0,1]

def create_train_data():
	train_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label=label_img(img)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		train_data.append([np.array(img),np.array(label)])
	shuffle(train_data)
	np.save('train_data.npy',train_data)
	return train_data

def process_test_data():
	test_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		path=os.path.join(TEST_DIR,img)
		img_number=img.split('.')[0]
		img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		test_data.append([np.array(img),img_number])
	shuffle(test_data)	
	np.save('test_data.npy',test_data)
	return test_data

train_data=create_train_data()
#if areadly created the dataset
#train_data=np.load('train_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded')

train=train_data[:-500]
test=train_data[-500:]

train_X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
train_Y=[i[1] for i in train]

test_X=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_Y=[i[1] for i in test]

model.fit({'input':train_X},{'targets':train_Y},n_epoch=5,validation_set=({'input':test_X},{'targets':test_Y}),
	snapshot_step=500,show_metric=True,run_id=MODEL_NAME)

model.save(MODEL_NAME)

test_data=process_test_data()
#if you alraady have test dataset
#test_data=np.load('test.data.npy')


def plot_image(row,col):
	for number,data in enumerate(test_data[:row*col]):
		img_number=data[1]
		img_data=data[0]
		f=plt.figure().add_subplot(row,col,number+1)
		orig=img_data
		data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)

		predictions=model.predict([data])[0]

		if np.argmax(predictions)==1: img_label='Dog'
		else: img_label='Cat'	

		f.imshow(orig,cmap=plt.cm.binary)
		plt.title(img_label)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()

plot_image(2,5)



