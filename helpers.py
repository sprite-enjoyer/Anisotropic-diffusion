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
  """
    Load grayscale images from a folder.

    Parameters:
    - path (str): Path to the folder containing images.
    - return_filenames (bool): If True, return a tuple with images and corresponding filenames. Default is False.

    Returns:
    - List of loaded images or a tuple (images, filenames) if return_filenames is True.

    Example:
    ```python
    images = get_images_from_path("base_images")
    ```
    """
  images = []

  for filename in os.listdir(path):
    img = cv2.cvtColor(cv2.imread(os.path.join(path,filename)), cv2.COLOR_BGR2GRAY)
    if img is not None:
      images.append(img)

  if return_filenames: 
    return [images, os.listdir(path)]
  
  return images


def convert_to_grayscale(source_path, save_path):
  """
    Convert color images to grayscale and save them.

    Parameters:
    - source_path (str): Path to the folder containing color images.
    - save_path (str): Path to the folder where grayscale images will be saved.

    Returns:
    None

    Example:
    ```python
    convert_to_grayscale("color_images", "grayscale_images")
    ```
    """
  images, filenames = get_images_from_path(source_path, return_filenames=True)
  save_images(images, save_path, filenames)


def save_images(images, folder_path, names=[]):
  """
    Save a list of images to a folder with specified filenames.

    Parameters:
    - images (list): List of images to be saved.
    - folder_path (str): Path to the folder where images will be saved.
    - names (list): List of filenames corresponding to the images. Default is an empty list.

    Returns:
    None

    Example:
    ```python
    save_images(images, "output_folder", ["image1.jpg", "image2.jpg"])
    ```
    """
  if len(names) != len(images): 
    raise ValueError("Length of the images must correspond to the length of the names")
  
  for i in range(len(images)): 
    integer_image = np.uint8(images[i])
    Image.fromarray(integer_image).save(folder_path + "/" + names[i])


def generate_noisy_images(source_path, save_path, salt_min=0.01, salt_max=0.3, pepper_min=0.01, pepper_max=0.3):
  """
  Generate noisy images by adding salt-and-pepper noise to grayscale images.

  Parameters:
  - source_path (str): Path to the folder containing the base grayscale images.
  - save_path (str): Path to the folder where the noisy images will be saved.
  - salt_min (float): Minimum salt noise probability. Default is 0.01.
  - salt_max (float): Maximum salt noise probability. Default is 0.3.
  - pepper_min (float): Minimum pepper noise probability. Default is 0.01.
  - pepper_max (float): Maximum pepper noise probability. Default is 0.3.

  Returns:
  None

  This function reads grayscale images from the source folder, applies salt-and-pepper noise
  to each image multiple times, and saves the resulting noisy images to the specified folder.

  Example:
  ```python
  generate_noisy_images("base_images", "noisy_images", salt_min=0.02, salt_max=0.2, pepper_min=0.01, pepper_max=0.25)
  ```
  """
  images = get_images_from_path(source_path)
  noisy_images = []
  salts = []
  peppers = []

  for i in range(len(images)):
    curr_image = images[i]
    curr_salts = []
    curr_peppers = []
    curr_noisy_images = []

    for _ in range(5):
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
