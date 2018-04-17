# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:48:20 2018

@author: Vikas
"""


import tensorflow as tf

session =tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(session.list_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)