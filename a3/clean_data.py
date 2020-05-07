import pickle
import numpy as np
from PIL import Image
import zipfile
import random

def load_pickle_file(path_to_file):
  """
  Loads the data from a pickle file and returns that object
  """
  ## Look up: https://docs.python.org/3/library/pickle.html 
  ## The code should look something like this:
  # with open(path_to_file, 'rb') as f:
  #   obj = pickle.... 
  ## We will let you figure out which pickle operation to use

  pkl_file = open(path_to_file, 'rb')
  data = pickle.load(pkl_file)
  pkl_file.close()
  return data


## You should define functions to resize, rotate and crop images
## below. You can perform these operations either on numpy arrays
## or on PIL images (read docs: https://pillow.readthedocs.io/en/stable/reference/Image.html)


def rotate(image, degrees):
  return image.rotate(degrees)

def resize(image):
  return image.resize((56, 56))



## We want you to clean the data, and then create a train and val folder inside
## the data folder (so your data folder in a3/ should look like: )
# data/
#   data.zip
#   train/
#   val/

## Inside the train and val folders, you will have to dump the CLEANED images and
## labels. You can dump images/annotations in a pickle file (because our data loader 
## expects the path to a pickle file.)

## Most code written in this file will be DIY. It's important that you get to practice
## cleaning datasets and visualising them, so we purposely won't give you too much starter
## code. It'll be up to you to look up documentation and understand different Python modules.
## That being said, the task shouldn't be too hard, so we won't send you down any rabbit hole.

if __name__ == "__main__":
  ## Running this script should read the input images.pkl and labels.pkl and clean the data
  ## and store cleaned data into the data/train and data/val folders
  
  ## To correct rotated images and add missing labels, you might want to prompt the terminal
  ## for input, so that you can input the angle and the missing label
  ## Remember, the first 60 images are rotated, and might contain missing labels.

  with zipfile.ZipFile('C:\\Users\\evely\\IntSys-Education\\a3\\data\\data.zip', 'r') as zip_ref:
    zip_ref.extractall()
  pkl_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\images.pkl', 'rb')
  data = pickle.load(pkl_file)
  label_pkl = open('C:\\Users\\evely\\IntSys-Education\\a3\\labels.pkl', 'rb')
  label_data = pickle.load(label_pkl)

  label_data[0][3] = 3
  label_data[1][1] = 2
  label_data[2][2] = 3
  label_data[3][0] = 5
  label_data[4][0] = 1
  label_data[5][1] = 1
  label_data[6][3] = 2
  label_data[7][3] = 3
  label_data[8][0] = 6
  label_data[9][3] = 2
  label_data[10][1] = 7
  label_data[11][2] = 7
  label_data[12][1] = 3
  label_data[13][2] = 6
  label_data[14][0] = 0

  resized = []
  for i in range(0, len(data)):
    resized.append(resize(data[i]))

  rotation_angles = [90,-45,180,-45,-90,-90,-45,90,180,90,315,180,-90,180,180]

  for i in range(0, len(rotation_angles)):
    resized[i] = rotate(resized[i], rotation_angles[i])

  data = resized

  index_shuffled = list(range(len(data)))
  random.shuffle(index_shuffled)

  train_indices = index_shuffled[:int(0.8*len(data))]
  val_indices = index_shuffled[int(0.8*len(data)):int(0.9*len(data))]
  test_indices = index_shuffled[int(0.9*len(data)):]

  train_labels = []
  train_samples = []

  for i in range(0, len(train_indices)):
    train_labels.append(label_data[train_indices[i]])
    train_samples.append(data[train_indices[i]])

  t_labels_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\train\\correct_train_labels.pkl', 'wb')
  pickle.dump(train_labels, t_labels_file)
  t_labels_file.close()

  t_samples_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\train\\train_samples.pkl', 'wb')
  pickle.dump(train_samples, t_samples_file)
  t_samples_file.close()

  val_labels = []
  val_samples = []

  for i in range(0, len(val_indices)):
    val_labels.append(label_data[val_indices[i]])
    val_samples.append(data[val_indices[i]])

  v_labels_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\val\\correct_val_labels.pkl', 'wb')
  pickle.dump(val_labels, v_labels_file)
  v_labels_file.close()

  v_samples_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\val\\val_samples.pkl', 'wb')
  pickle.dump(val_samples, v_samples_file)
  v_samples_file.close()

  test_labels = []
  test_samples = []

  for i in range(0, len(test_indices)):
    test_labels.append(label_data[test_indices[i]])
    test_samples.append(data[test_indices[i]])

  te_labels_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\test\\correct_test_labels.pkl', 'wb')
  pickle.dump(test_labels, te_labels_file)
  te_labels_file.close()

  te_samples_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\test\\test_samples.pkl', 'wb')
  pickle.dump(test_samples, te_samples_file)
  te_samples_file.close()
