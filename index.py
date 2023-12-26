from anisotropic_diffusion import anisotropic_diffusion as anisodiff
from helpers import *

def exec(base_path, grayscale_path, noisy_path, result_path, noisy_generated=True, grayscale_generated=True):
  if not grayscale_generated: 
    convert_to_grayscale(base_path, grayscale_path)
  if not noisy_generated: 
    generate_noisy_images(grayscale_path, noisy_path)

  images, file_names = get_images_from_path(noisy_path, return_filenames=True)
  diffused_images = [anisodiff(image, kappa=80, num_iterations=2, methodNum=1) for image in images]
  save_images(diffused_images, result_path, file_names)

exec(
  "base_images", 
  "grayscale_images", 
  "noisy_images", 
  "result_images", 
  noisy_generated=True, 
  grayscale_generated=True
)