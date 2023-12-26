import numpy as np
import os
import cv2
import random
from PIL import Image

def add_spn(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Add salt-and-pepper noise to an image.

    Parameters:
    - image: Input image.
    - salt_prob: Probability of adding salt noise.
    - pepper_prob: Probability of adding pepper noise.

    Returns:
    - Noisy image.
    """
    noisy_image = np.copy(image)

    # Add salt noise
    salt = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt] = 255

    # Add pepper noise
    pepper = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper] = 0

    return noisy_image

def get_images_from_path(path, return_filenames=False):
  images = []

  for filename in os.listdir(path):
    img = cv2.cvtColor(cv2.imread(os.path.join(path,filename)), cv2.COLOR_BGR2GRAY)
    if img is not None:
      images.append(img)

  if return_filenames: 
    return [images, os.listdir(path)]
  
  return images

def convert_to_grayscale(source_path, save_path):
  images, filenames = get_images_from_path(source_path, return_filenames=True)
  save_images(images, save_path, filenames)

def save_images(images, folder_path, names=[]):
  if len(names) != len(images): 
    raise ValueError("Length of the images must correspond to the length of the names")
  
  for i in range(len(images)): 
    integer_image = np.uint8(images[i])
    Image.fromarray(integer_image).save(folder_path + "/" + names[i])

def generate_noisy_images(source_path, save_path, salt_min=0.01, salt_max=0.2, pepper_min=0.01, pepper_max=0.2):
  images = get_images_from_path(source_path)
  noisy_images = []
  salts = []
  peppers = []

  for i in range(len(images)):
    curr_image = images[i]
    curr_salts = []
    curr_peppers = []
    curr_noisy_images = []

    for _ in range(3):
      salt = round(random.uniform(salt_min, salt_max), 2)
      pepper = round(random.uniform(pepper_min, pepper_max), 2)
      noisy_image = add_spn(curr_image, salt, pepper)
      curr_noisy_images.append(noisy_image)
      curr_salts.append(salt)
      curr_peppers.append(pepper)

    salts.append(curr_salts)
    peppers.append(curr_peppers)
    noisy_images.append(curr_noisy_images)

  for i in range(len(noisy_images)):
    curr_image = noisy_images[i]
    for j in range(len(curr_image)):
      curr_salt = salts[i][j]
      curr_pepper = peppers[i][j]
      noisy_curr = curr_image[j]
      image = Image.fromarray(noisy_curr)
      path = save_path + "/" + str(i) + "_" + "s=" + str(curr_salt) + "_" + "p=" + str(curr_pepper) + ".jpg"
      image.save(path)
