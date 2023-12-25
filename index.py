from PIL import Image
import cv2
import numpy as np

class numerical_methods:

  """
  class description goes here
  """

  @staticmethod
  def get_numerical_method(selected_method):
    if (selected_method != 1 and selected_method !=2):
      raise ValueError("1 - Adams-Bashforth, 2 - Runge-Kutta, Other values are invalid!")
    
    match selected_method: 
      case 1: return numerical_methods.adams_bashforth_step
      case 2: return numerical_methods.runge_kutta_step

  @staticmethod
  def adams_bashforth_step(image, f, gamma):
    return image + gamma * f(image)
  
  @staticmethod
  def runge_kutta_step(image, f, gamma):
    k1 = gamma * f(image)
    k2 = gamma * f(image + 0.5 * k1)
    k3 = gamma * f(image + 0.5 * k2)
    k4 = gamma * f(image + k3)
    return image + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def anisotropic_diffusion(initial_image, num_iterations=50, kappa=10, gamma=0.1, methodNum=1):
    """
    Perform anisotropic diffusion using different numerical methods.

    Parameters:
    - image: Input image.
    - num_iterations: Number of iterations.
    - kappa: Diffusion coefficient.
    - gamma: Time step.
    - method: Numerical method to use:
      method = 1: Adams-Moulton
      method = 2: Adams-Bashforth
      method = 3: Runge-Kutta
      method = 4: Diagonally Implicit Runge-Kutta

    Returns:
    - diffused_image: Diffused image.
    """
    numerical_method = numerical_methods.get_numerical_method(methodNum)
    initial_image = initial_image.astype('float32')    # To avoid np type errors
    result_image = initial_image.copy()                # Create result image variable
    vertical_gradient = np.zeros_like(result_image)    # Create vertical gradient which has the shape of the image
    horizontal_gradient = vertical_gradient.copy()     # Create the horizontal gradient in the same manner
    north_shifted = vertical_gradient.copy()           # Create varable that is currently identical to vertical_gradient
    west_shifted = vertical_gradient.copy()            # The same as above
    conduction_vertical = np.ones_like(result_image)   # Array of the shape of result_image, filled with ones
    conduction_horizontal = conduction_vertical.copy() # The copy of the above array

    def update_conduction_arrs(h, v): 
      v = np.exp(-(vertical_gradient / kappa) ** 2.)
      h = np.exp(-(horizontal_gradient / kappa) ** 2.)

    for _ in range(num_iterations):      

      # Computes the difference between element ii and its previous element i on vertical axis
      vertical_gradient[:-1, :] = np.diff(result_image, axis=0)
      # Does the same as above but on horizontal axis
      horizontal_gradient[:, :-1] = np.diff(result_image, axis=1)
      
      update_conduction_arrs(conduction_horizontal, conduction_vertical)
      f = lambda y: y + gamma * (north_shifted + west_shifted)
      result_image = numerical_method(result_image, f, gamma)
      
      h = conduction_horizontal * horizontal_gradient
      v = conduction_vertical * vertical_gradient

      north_shifted[:] = v
      west_shifted[:] = h
      north_shifted[1:, :] -= v[:-1,:]
      west_shifted[:, 1:] -= h[:,:-1]

    return result_image

image = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2GRAY)

test1_image = anisotropic_diffusion(image, num_iterations=10, methodNum=2)
final_image = Image.fromarray(test1_image)
final_image.show()