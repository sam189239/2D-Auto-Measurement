import numpy as np
import os
import sys
import tarfile
from six.moves import urllib
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse
import tensorflow as tf


#### Segmentation ####

class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		#"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.compat.v1.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
	A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
		  colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap

def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
	label: A 2D array with integer type, storing the segmentation label.

	Returns:
	result: A 2D array with floating type. The element of the array
	  is the color indexed by the corresponding element in the input label
	  to the PASCAL color map.

	Raises:
	ValueError: If label is not of rank 2 or its value is larger than color
	  map maximum entry.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]


def segmentation_model():
  LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])

  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


  MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
      'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
      'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
      'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
      'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
  }
  _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

  model_dir = 'deeplab_model'
  if not os.path.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

  download_path = os.path.join(model_dir, _TARBALL_NAME)
  if not os.path.exists(download_path):
    print('downloading model to %s, this might take a while...' % download_path)
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], 
            download_path)
    print('download completed! loading DeepLab model...')

  MODEL = DeepLabModel(download_path)
  print('model loaded successfully!')
  return MODEL

def bg_removal(image, seg):
  seg=cv2.resize(seg.astype(np.uint8),image.size)
  mask_sel=(seg==15).astype(np.float32)
  mask = 255*mask_sel.astype(np.uint8)

  img = 	np.array(image)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   

  res = cv2.bitwise_and(img,img,mask = mask)
  bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)) 
  plt.subplot(131)
  plt.title('OG image')
  plt.imshow(image)
  plt.axis('off')
  plt.subplot(132)
  plt.title('Seg Mask')
  plt.imshow(seg)
  plt.axis('off')
  plt.subplot(133)
  plt.title('BG removed')
  plt.imshow(bg_removed)
  plt.axis('off')
  plt.savefig("..\\..\\out\\seg_op.jpg")
  plt.show()
  cv2.imwrite("..\\..\\out\\bg_rem_img.jpg",bg_removed)
  cv2.imwrite("..\\..\\out\\seg_mask.jpg",mask)
  return mask, bg_removed 


#### hmr 2.0 ####

import sys
print(sys.executable, sys.version)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import trimesh

from os.path import join, abspath
from os import mkdir
from IPython.display import display, HTML
from glob import glob
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# %matplotlib inline

# for local import 
sys.path.append(abspath('..'))

from main.config import Config
from main.model import Model
from main.dataset import Dataset
from main.smpl import Smpl
from main.local import LocalConfig

from visualise.vis_util import draw_2d_on_image, show_2d_pose, show_3d_pose, preprocess_image, resize_img, visualize

def HMRmodel():
  class TrimeshConfig(LocalConfig):
          BATCH_SIZE = 1
          ENCODER_ONLY = True
          LOG_DIR = abspath('..\\..\\logs\\paired\\base_model')
          
  config = TrimeshConfig()

  # inizialize model 
  model = Model()
  return model

def hmr(inp, model):
  input_frames = [inp]
  results = []
  joints = []
  vertices = []
  img_2ds = []
  cams = []

  for image in input_frames:
      result = model.detect(image)
      results.append(result)
      joint = np.squeeze(result['kp2d'].numpy())
      joints.append(joint)
      np.save('..\\..\\out\\joints.npy',joints)
      img_2ds.append(draw_2d_on_image(image, joint))
      cams.append(np.squeeze(result['cam'].numpy())[:3])
      vertices.append(np.squeeze(result['vertices'].numpy()))

  # f, ax = plt.subplots(1,2)
  # ax[0].imshow(img_2ds[0])
  # ax[1].imshow(img_2ds[1])

  plt.imshow(img_2ds[0])
  cv2.imwrite("..\\..\\out\\joints_plot.jpg",img_2ds[0])
  # f.set_size_inches(20,20)
  return joints, vertices, cams


### Feature Points ###

def dist(a,b,c,d):
  return ((d-b)**2 + (c-a)**2)**0.5

def neck_pts(mask,n_p):
  vlid_r = False
  valid_l = False
  for a in range(int(n_p[0]),np.shape(mask)[1]):
    if(mask[int(n_p[1])][a]) == 0:
      right = a
      valid_r = True
      break
  for a in range(int(n_p[0]))[::-1]:
    if(mask[int(n_p[1])][a]) == 0:
      left = a
      valid_l = True
      break
  if(not(valid_r) or not(valid_l)):
    print("Invalid")
    return 0,0
  return right,left

def ht_pts(seg,f):
  temp = sum(np.transpose(seg))
  valid_top = False
  valid_bot = False
  factor = f / 100 * np.shape(seg)[1]
  for a in (range(len(temp))):
    if temp[a] > factor:
      top = a 
      valid_top = True
      break
  temp = temp[::-1]
  for a in (range(len(temp))):
    if temp[a] > factor:
      bottom = np.shape(seg)[0]-a
      valid_bot = True
      break
  mid = int((bottom - top)/2) + top
  if(not(valid_top) or not(valid_bot)):
    print("Invalid")
    return 0,0

  temp = seg
#   temp[[top,bottom,mid],:] = 1

  # plt.imshow(temp)
  # plt.show()
  return top,bottom

def convert_to_int(image, joints):
    # convert image float to int
    if image.min() < 0. and image.max() < 2.:
        image = ((image + 1) / 2 * 255).astype(np.uint8)

    # convert joints float to int
    if np.issubdtype(joints.dtype, np.floating):
        if joints.min() < 0. and joints.max() < 2.:
            joints = _convert_joints(joints, image.shape[:2])
        else:
            joints = np.round(joints).astype(np.int32)

    return image, joints

def _convert_joints(joints, img_shape):
    return ((joints + 1) / 2 * img_shape).astype(np.int32)

def preprocess_gray(mask):
  img_size = 224
  scale = (float(img_size) / np.max(mask.shape[:2]))
  mask_scaled, actual_factor = resize_img(mask, scale)
  center = np.round(np.array(mask_scaled.shape[:2]) / 2).astype(int)
  center = center[::-1]  # image center in (x,y)

  margin = int(img_size / 2)
  image_pad = np.pad(mask_scaled, ((margin,), (margin,)), mode='edge')
  center_pad = center + margin
  start = center_pad - margin
  end = center_pad + margin

  crop = image_pad[start[1]:end[1], start[0]:end[0]]
  return crop


def shift_neck(joints, neck_shift_factor, ht_p):
  return np.array([joints[12][0], joints[12][1]-ht_p*neck_shift_factor])


def corner_detection(image):
  # operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  operatedImage = image
  operatedImage = np.float32(operatedImage)  
  dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 
  # Results are marked through the dilated corners
  dest = cv2.dilate(dest, None)
  # Reverting back to the original image,
  # with optimal threshold value
  out = np.zeros((np.shape(image)))
  dest[dest > 0.01 * dest.max()]=[255]
  # the window showing output image with corners
  # cv2.imshow('Image with Borders', dest)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # plt.imshow(dest)
  return dest

def shortest_neck(dest_crop, sn_p):
    min_r, min_l = 10000, 10000
    r,l = [0,0],[0,0]
    if len(dest_crop)==0:
        print("Invalid crop")
        return 0,0
    for a in range(len(dest_crop)):
        for b in range(len(dest_crop[0])):
            if abs(dest_crop[a][b]) > 0:
                if b <= sn_p[0]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_l:
                        min_l = dist(a,b,sn_p[0],sn_p[1])
                        l = [a,b]
                else:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_r:
                        min_r = dist(a,b,sn_p[0],sn_p[1])
                        r = [a,b]
    return r,l

def shift_waist(joints, ht_p, waist_shift_factor):
  return np.array([joints[2][0]/2 + joints[3][0]/2, joints[2][1]/2 + joints[3][1]/2 - ht_p*waist_shift_factor])

def waist_pts(mask,n_p):
  valid_r = False
  valid_l = False
  for a in range(int(n_p[0]),np.shape(mask)[0]):
    if(mask[int(n_p[1])][a]) == 0:
      right = a
      valid_r = True
      break
  for a in range(int(n_p[0]))[::-1]:
    if(mask[int(n_p[1])][a]) == 0:
      left = a
      valid_l = True
      break
  if(not(valid_r) or not(valid_l)):
    print("Invalid")
    return 0,0
  return right,left


def shortest_hand(dest_crop, sn_p):
    min_r, min_l = 10000, 10000
    r,l = [0,0],[0,0]
    if len(dest_crop)==0:
        print("Invalid crop")
        return 0,0
    for a in range(len(dest_crop)):
        for b in range(len(dest_crop[0])):
            if abs(dest_crop[a][b]) > 0:
                if a <= sn_p[1]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_l:
                        min_l = dist(a,b,sn_p[0],sn_p[1])
                        l = [a,b]
                else:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_r:
                        min_r = dist(a,b,sn_p[0],sn_p[1])
                        r = [a,b]
    return r,l


def shortest_hand_right(dest_crop, sn_p):
    min_r, min_l = 10000, 10000
    r,l = [0,0],[0,0]
    if len(dest_crop)==0:
        print("Invalid crop")
        return 0,0
    for a in range(len(dest_crop)):
        for b in range(len(dest_crop[0])):
            if abs(dest_crop[a][b]) > 0:
                if a <= sn_p[1] and b <= sn_p[0]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_l:
                        min_l = dist(a,b,sn_p[0],sn_p[1])
                        l = [a,b]
                elif a > sn_p[1] and b > sn_p[0]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_r:
                        min_r = dist(a,b,sn_p[0],sn_p[1])
                        r = [a,b]
    return r,l


def shortest_hand_left(dest_crop, sn_p):
    min_r, min_l = 10000, 10000
    r,l = [0,0],[0,0]
    if len(dest_crop)==0:
        print("Invalid crop")
        return 0,0
    for a in range(len(dest_crop)):
        for b in range(len(dest_crop[0])):
            if abs(dest_crop[a][b]) > 0:
                if a > sn_p[1] and b <= sn_p[0]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_l:
                        min_l = dist(a,b,sn_p[0],sn_p[1])
                        l = [a,b]
                elif a <= sn_p[1] and b > sn_p[0]:
                    if dist(a,b,sn_p[0],sn_p[1])<=min_r:
                        min_r = dist(a,b,sn_p[0],sn_p[1])
                        r = [a,b]
    return r,l


  ### Measurements ###

def front_measurement(joints, r_waist, l_waist, r_neck, l_neck, lpp, arm_scale):
  shoulder = dist(joints[8][0],joints[8][1],joints[9][0],joints[9][1])*lpp
  arm = ((dist(joints[8][0],joints[8][1],joints[7][0],joints[7][1])+dist(joints[7][0],joints[7][1],joints[6][0],joints[6][1])
        +dist(joints[9][0],joints[9][1],joints[10][0],joints[10][1])+dist(joints[10][0],joints[10][1],joints[11][0],joints[11][1]))/2)*lpp * arm_scale
  waist_f = dist(r_waist[0],r_waist[1],l_waist[0],l_waist[1])*lpp
  neck_f = dist(r_neck[0],r_neck[1],l_neck[0],l_neck[1])*lpp
  print("Shoulder: "+str(shoulder)+" cm")
  print("Arm: "+str(arm)+" cm")
  print("Waist Front: "+str(waist_f)+" cm")
  print("Neck Front: "+str(neck_f)+" cm")
  return waist_f, neck_f

def side_measurement(joints, r_waist, l_waist, r_neck, l_neck, lpp):
  waist_s = dist(r_waist[0],r_waist[1],l_waist[0],l_waist[1])*lpp
  neck_s = dist(r_neck[0],r_neck[1],l_neck[0],l_neck[1])*lpp
  print("Waist Side: "+str(waist_s)+" cm")
  print("Neck Side: "+str(neck_s)+" cm")
  return waist_s, neck_s

def circumference(name, a, b):
  ans = 2 * 3.14159 * ((a**2+b**2)/2)**0.5
  print(name + " circumference: "+str(ans)+" cm")