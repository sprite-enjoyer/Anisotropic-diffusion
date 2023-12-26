# Anisotropic Diffusion (Perona-Malik) Codebase

This codebase provides a simple implementation of the Anisotropic Diffusion algorithm, specifically the Perona-Malik version. The algorithm is used for enhancing images by reducing noise while preserving important edges.

## Usage Instructions

To use the Anisotropic Diffusion algorithm on noisy images, follow these steps:

### Step 1: Prepare Noisy Images

Place the noisy images you want to process in the `noisy_images` folder.

### Step 2: Configure Execution

Modify the arguments of the `exec` function in `index.py` to specify the input and output folders. Example:

```python
exec(
  "base_images",
  "grayscale_images",
  "noisy_images",
  "result_images",
  noisy_generated=True,
  grayscale_generated=True
)
```

### Step 3: Execute the Algorithm

Run the modified `index.py` script. The Anisotropic Diffusion algorithm will be applied to the noisy images.

### Step 4: View Results

After the execution, the resulting enhanced images will be available in the `result_images` folder.

Note: If `grayscale_generated` is set to `True`, base grayscale images will be generated from color images.

Feel free to experiment with parameters such as `kappa`, `num_iterations`, and `methodNum` in the `anisodiff` function for different effects.
