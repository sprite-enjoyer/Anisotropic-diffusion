from anisotropic_diffusion import anisotropic_diffusion as ad
from helpers import *
from tqdm import tqdm

def exec(base_path, grayscale_path, noisy_path, result_path, noisy_generated=True, grayscale_generated=True):
  if not grayscale_generated: 
    convert_to_grayscale(base_path, grayscale_path)
  if not noisy_generated: 
    generate_noisy_images(grayscale_path, noisy_path)

  images, file_names = get_images_from_path(noisy_path, return_filenames=True)
  diffused_images = [ad(image, kappa=100, num_iterations=0) for image in images]
  save_images(diffused_images, result_path, file_names)

exec(
  "base_images", 
  "grayscale_images", 
  "noisy_images", 
  "result_images", 
  noisy_generated=True, 
  grayscale_generated=True
)

def exec_for_analysis(base_path, grayscale_path, noisy_path, result_path, noisy_generated=True, grayscale_generated=True):
  if not grayscale_generated: 
    convert_to_grayscale(base_path, grayscale_path)
  if not noisy_generated: 
    generate_noisy_images(grayscale_path, noisy_path)

  images, file_names = get_images_from_path(noisy_path, return_filenames=True)
  diffused_images = []
  names = []

  for i in range(len(images)): 
    for kappa in tqdm(range(1, 100, 5)):
      for num_iterations in tqdm(range(1, 5, 1)):
        for method in range(1):
          res = ad(images[i], kappa=kappa, num_iterations=num_iterations, methodNum=method + 1)
          diffused_images.append(res)
          print(file_names[i])
          new_filename = file_names[i][:-4]+"_k="+str(kappa)+"_i="+str(num_iterations)+"_m="+str(method)+".jpg"
          names.append(new_filename)

  save_images(diffused_images, result_path, names)

# exec_for_analysis(
#   "base_images", 
#   "grayscale_images", 
#   "noisy_images", 
#   "analysis_images", 
#   noisy_generated=True, 
#   grayscale_generated=True
# )