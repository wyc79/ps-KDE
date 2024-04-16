from PIL import Image
import numpy as np
import os
import pandas as pd
import tqdm
import scipy.stats as stats


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

def get_files_normalized(paths, loc, sub, size):
  mask = Image.open(os.path.join(paths[loc], f"{sub}.png"))
  mask = np.array(mask.resize(size), dtype="float")
  mask[mask!=0] = 1
  mask[mask==0] = 0 # just to make it look symmetric :)
  raw = Image.open(os.path.join(paths["images"], f"{sub}.png"))
  raw = np.array(raw.resize(size))/255.
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

  ### calculate the kde dataframe
  step = np.round(0.1**digits, digits)
  df = pd.DataFrame(np.ones((int(1/step)+1, len(locs)+1)), 
                    index=np.round(np.arange(0,1+step,step), digits), 
                    columns=locs+['all'])

  for loc in locs:
    print(f"Loading location: {loc}")
    for sub in tqdm.tqdm(subs):
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
  print(f"Calculating KDE:")
  for col in tqdm.tqdm(cols):
    df[col] = df[col] / np.sum(df[col])
    resamples = np.random.choice(df.index.values, size=1000000, p=df[col])
    rkde = stats.gaussian_kde(resamples)
    data_space = df.index.values
    evaluated = rkde.evaluate(data_space)
    df[col] = evaluated / np.sum(evaluated)

  if save_path is not None:
    df.to_csv(save_path)

  return df

def map_from_df(image, df, loc, digits=3):
  # map the image based on provided df
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
