import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch import optim
from torch.autograd import Variable 
from torch.utils import data
from PIL import Image
from matplotlib.pyplot import imshow
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import random
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import tqdm




def get_files(paths, loc, sub, size, normalize=False, img_fmt="png", mask_fmt="png"):
  mask = Image.open(os.path.join(paths[loc], f"{sub}.{mask_fmt}"))
  mask = np.array(mask.resize(size), dtype="float")
  mask[mask!=0] = 1
  mask[mask==0] = 0 # just to make it look symmetric :)
  raw = Image.open(os.path.join(paths["images"], f"{sub}.{img_fmt}"))
  raw = np.array(raw.resize(size))
  if normalize:
    raw = raw/255.
  if len(raw.shape) == 2:
    raw = np.stack((raw,)*3, axis=-1)
  else:
    raw = raw[:,:,0:3]
  # print(os.path.join(paths["images"], f"{sub}.png"))
  return raw, mask

def get_label_pixel(raw, mask):
  """
  returns the pixel values where the mask label is 1
  @param raw: original image pixel values with shape (dim, dim, 3)
         mask: labels whether a pixel is 1 or not with shape(dim, dim)

  @return matrix: 2-D matrix with size shape(mask). 0 if not in label, pixel_val if in label.
  """
  raw_2d_shape = raw.shape[0:2]

  # check if images are the same dimension
  assert raw_2d_shape == mask.shape, f"raw shape {raw_2d_shape} does not match mask shape {mask.shape}"
  
  raw_2d = raw[:, :, 0] # only want the first channel
  
  matrix = raw_2d * mask # overlay labels with pixel values

  assert matrix.shape == mask.shape, f"matrix shape {matrix.shape} does not match mask shape {mask.shape}"

  return matrix

def generate_df(paths, locs, subs, size=(256,256), digits=3, save_path=None):
  step = np.round(0.1**digits, digits)
  df = pd.DataFrame(np.ones((int(1/step)+1, len(locs)+1)), 
                    index=np.round(np.arange(0,1+step,step), digits), 
                    columns=locs+['all'])

  for loc in locs:
    print(f"Location: {loc}")
    for sub in tqdm.tqdm(subjects):
      raw3d, mask = get_files_normalized(paths, loc, sub, size)
      label_pixel_val = get_label_pixel(raw3d, mask)
      all = np.reshape(raw3d, [-1])
      all = np.round(all[all>0], digits)
      masked = np.reshape(label_pixel_val, [-1])
      masked = np.round(masked[masked>0], digits)

      unique_m, counts_m = np.unique(masked, return_counts=True)
      for i,px in enumerate(unique_m):
        df.loc[px, loc] = df.loc[px, loc] + counts_m[i]
      
      unique_a, counts_a = np.unique(all, return_counts=True)
      for i,px in enumerate(unique_a):
        df.loc[px, "all"] = df.loc[px, "all"] + counts_a[i]
  
  cols = df.columns
  for col in cols:
    df[col] = df[col] / np.sum(df[col])

  if save_path is not None:
    df.to_csv(save_path)

  return df

def map_from_df(image, df, loc, digits=3):
  out = image.copy()
  image = np.round(image, digits)
  for i in df.index:
    val = df.loc[i, loc]
    out = np.where(image==i, val, out)
  out = (out - np.min(out)) / (np.max(out) - np.min(out))
  return out

def preprocessing(file, df, loc, size=(256,256)):
  raw = file / 255.
  if len(raw.shape) == 3:
    raw = raw[:,:,0]
  out = map_from_df(raw, df, loc)
  out = np.stack((out,)*3, axis=-1)
  return out


def image_aug_generator(location, seed, train, img_path=None, mask_path=None, df=None, windowing=False, size=(256,256)):
  '''
  Input:  location: heart, left lung, right lung, left clavicle, right clavicle
  Return: generator for image and its corresponding label for the specified anatomical location
  '''
  if (img_path is None) & (mask_path is None):
    if train:
      img_path = Working_dir + "SCR_Data/split_output/train/images/" 
      mask_path = Working_dir + "SCR_Data/split_output/train/" + location + "/"
    else:
      img_path = Working_dir + "SCR_Data/split_output/test/images/" 
      mask_path = Working_dir + "SCR_Data/split_output/test/" + location + "/"
    print(img_path)
    print(mask_path)

  # Random augmentation parameters
  data_gen_args = dict(rotation_range=20,
                      width_shift_range=0.15,
                      height_shift_range=0.15,
                      zoom_range=[0.8, 1.2],
                      rescale=1./255)
  
  if (windowing==True) & (df is not None):
    pp = lambda file : preprocessing(file, df, location, size=size)
    image_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=pp)
    # image_datagen = ImageDataGenerator(**data_gen_args)

  else:
    image_datagen = ImageDataGenerator(**data_gen_args)

  mask_datagen = ImageDataGenerator(**data_gen_args)

  # Provide the same seed and keyword arguments to the fit and flow methods
  # seed = seed
  
  # generators for image and mask
  image_generator = image_datagen.flow_from_directory(
      img_path,
      target_size=size,
      batch_size=batch_size,
      class_mode=None,
      seed=seed)
  mask_generator = mask_datagen.flow_from_directory(
      mask_path,
      target_size=size,
      batch_size=batch_size,
      class_mode=None,
      seed=seed)
  
  # combine generators into one
  # train_generator = zip(image_generator, mask_generator)
  my_generator = my_image_mask_generator(image_generator, mask_generator)


  return my_generator
  # return image_generator, mask_generator



def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


