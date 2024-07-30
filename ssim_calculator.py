import argparse
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Takes input from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('-f1', '--file1', required=True, help='First image file')
parser.add_argument('-f2', '--file2', required=True, help='Second image file')
args = parser.parse_args()

filename1 = args.file1
filename2 = args.file2

# If files are not specified as .jpg, add .jpg to the end
if not filename1.endswith('.jpg'):
    filename1 += '.jpg'
if not filename2.endswith('.jpg'):
    filename2 += '.jpg'

# Read images
image1 = imread(filename1)
image2 = imread(filename2)

# Convert images to grayscale
image1_gray = rgb2gray(image1)
image2_gray = rgb2gray(image2)

# Determine the smallest dimensions between the two images
min_height = min(image1_gray.shape[0], image2_gray.shape[0])
min_width = min(image1_gray.shape[1], image2_gray.shape[1])

# Resize images to the smallest dimensions
image1_resized = resize(image1_gray, (min_height, min_width), anti_aliasing=True)
image2_resized = resize(image2_gray, (min_height, min_width), anti_aliasing=True)

# Print out the dimensions of each resized image
print(f'Updated dimensions of {filename1}: {image1_resized.shape}')
print(f'Updated dimensions of {filename2}: {image2_resized.shape}')

crop_x_start = 223
crop_x_end = 1880
image1_cropped = image1_resized[:, crop_x_start:crop_x_end]
image2_cropped = image2_resized[:, crop_x_start:crop_x_end]

# Pixelate images with specified block size
block_sizex = 103
block_sizey = 101

def pixelate_image(image, block_sizex, block_sizey):
    height, width = image.shape
    pixelated_image = np.zeros((height, width))
    
    for i in range(0, height, block_sizey):
        for j in range(0, width, block_sizex):
            block = image[i:i+block_sizey, j:j+block_sizex]
            if block.size > 0:
                average_color = block.mean()
                pixelated_image[i:i+block_sizey, j:j+block_sizex] = average_color
    
    return pixelated_image

# Pixelate images
image1_pixelated = pixelate_image(image1_cropped, block_sizex, block_sizey)
image2_pixelated = pixelate_image(image2_cropped, block_sizex, block_sizey)

# Define the grid size
num_grids_x = 16
num_grids_y = 38

# Calculate SSIM for each grid block
ssim_map = np.zeros((num_grids_y, num_grids_x))
ssim_values = []

for i in range(num_grids_y):
    for j in range(num_grids_x):
        x_start = j * block_sizex
        x_end = min((j + 1) * block_sizex, image1_cropped.shape[1])
        y_start = i * block_sizey
        y_end = min((i + 1) * block_sizey, image1_cropped.shape[0])
        
        block1 = image1_cropped[y_start:y_end, x_start:x_end]
        block2 = image2_cropped[y_start:y_end, x_start:x_end]
        
        if block1.size > 0 and block2.size > 0:
            ssim_val = ssim(block1, block2, data_range=1.0)
            ssim_map[i, j] = ssim_val
            ssim_values.append(ssim_val)

# Increase font size for all text elements
plt.rcParams.update({'font.size': 14})

# Plotting the images, SSIM heatmap, and 1D histogram
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5], width_ratios=[0.7, 0.7, 0.8])

# Plot Image 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image1_pixelated, cmap='gray')
ax1.set_xlabel('Pixels', fontsize=16)
ax1.set_ylabel('Pixels', fontsize=16)
ax1.axis('on')
ax1.tick_params(axis='both', which='major', labelsize=14)

# Plot Image 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(image2_pixelated, cmap='gray')
ax2.axis('on')
ax2.set_xlabel('Pixels', fontsize=16)
ax2.set_ylabel('Pixels', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Plot SSIM heatmap
ax3 = fig.add_subplot(gs[0, 2])
cax = ax3.imshow(ssim_map, cmap='Blues', interpolation='nearest', vmin=0.7, vmax=1.0)
ax3.set_title('SSIM Comparison', fontsize=18)
ax3.axis('off')
cbar = fig.colorbar(cax, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14)

# Plot 1D histogram of SSIM values
ax4 = fig.add_subplot(gs[1, :])
ax4.hist(ssim_values, bins=20, color='grey', edgecolor='black')
ax4.set_xlim([0.6, 1])
ax4.set_title('SSIM Values Histogram', fontsize=18)
ax4.set_xlabel('SSIM Value', fontsize=16)
ax4.set_ylabel('Frequency', fontsize=16)
ax4.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()
