import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from config_reader import config_reader
import scipy
import math
import os
import json

def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x):
     
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)    
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)    
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)    
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)
    
    return x

def stage1_block(x, num_p, branch):
    
    # Block 1        
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    
    return x

def stageT_block(x, num_p, stage, branch):
        
    # Block 1        
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    
    return x

weights_path = "keras/model.h5" # orginal weights converted from caffe
#weights_path = "training/weights.best.h5" # weights tarined from scratch 

input_shape = (None,None,3)

img_input = Input(shape=input_shape)

stages = 6
np_branch1 = 38
np_branch2 = 19

img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

# VGG
stage0_out = vgg_block(img_normalized)

# stage 1
stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

# stage t >= 2
for sn in range(2, stages + 1):
    stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
    stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
model.load_weights(weights_path)
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util

extensions_img = {".jpg", ".png", ".gif", ".bmp", ".jpeg"}

for filename in os.listdir('sample_images'):
  for ext in extensions_img:
    if filename.endswith(ext):
      test_image = 'sample_images/'+filename
      file_noExt = os.path.splitext(filename)[0]
      print('Now proccessing:', filename)
    
      oriImg = cv2.imread(test_image) # B,G,R order
      param, model_params = config_reader()

      multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

      heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
      paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))


      for m in range(len(multiplier)):
          scale = multiplier[m]
          imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
          imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        

          input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
          print("Input shape: " + str(input_img.shape))  

          output_blobs = model.predict(input_img)
          print("Output shape (heatmap): " + str(output_blobs[1].shape))

          # extract outputs, resize, and remove padding
          heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
          heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
          heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
          heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

          paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
          paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
          paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
          paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

          heatmap_avg = heatmap_avg + heatmap / len(multiplier)
          paf_avg = paf_avg + paf / len(multiplier)


      from numpy import ma
      U = paf_avg[:,:,16] * -1
      V = paf_avg[:,:,17]
      X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
      M = np.zeros(U.shape, dtype='bool')
      M[U**2 + V**2 < 0.5 * 0.5] = True
      U = ma.masked_array(U, mask=M)
      V = ma.masked_array(V, mask=M)

      from scipy.ndimage.filters import gaussian_filter
      all_peaks = []
      peak_counter = 0

      for part in range(19-1):
          map_ori = heatmap_avg[:,:,part]
          map = gaussian_filter(map_ori, sigma=3)

          map_left = np.zeros(map.shape)
          map_left[1:,:] = map[:-1,:]
          map_right = np.zeros(map.shape)
          map_right[:-1,:] = map[1:,:]
          map_up = np.zeros(map.shape)
          map_up[:,1:] = map[:,:-1]
          map_down = np.zeros(map.shape)
          map_down[:,:-1] = map[:,1:]

          peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
          peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
          peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
          id = range(peak_counter, peak_counter + len(peaks))
          peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

          all_peaks.append(peaks_with_score_and_id)
          peak_counter += len(peaks)

      json_template = '{"version":1.2,"people":[{"pose_keypoints":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}'

      all_peaks_list = []

      for element in all_peaks:
        while len(element) > 1:
          element.pop()

      for idx, val in enumerate(all_peaks):
        if len(val) < 1:
          all_peaks_list.append([0,0,0,0])
        elif len(val) >= 1:
          all_peaks_list.append(list(val[0]))
        if idx == 17:
          break

      for i in all_peaks_list:
        i.pop()

      all_peaks_list.insert(8, [(x+y)/2.0 for (x, y) in zip(all_peaks_list[8], all_peaks_list[11])])

      for i in range(6):
        all_peaks_list.append([0,0,0])

      all_peaks_flat = [item for sublist in all_peaks_list for item in sublist]

      all_peaks_flat = [float(i) for i in all_peaks_flat]

      o = json.loads(json_template)

      o['people'][0]['pose_keypoints'] = all_peaks_flat


      with open('sample_jsons/'+file_noExt+'.json', 'w') as outfile:
          json.dump(o, outfile)
