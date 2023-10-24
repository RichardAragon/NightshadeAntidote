# Importing the required libraries
import cv2 # For image processing
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting and visualization
import scipy.fftpack as fft # For Fourier Transform
from PIL import Image # For metadata extraction
from collections import Counter # For histogram calculation

# Loading and preprocessing the input image
img = cv2.imread("input.jpg") # Reading the image from file
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting to grayscale
img = cv2.resize(img, (512, 512)) # Resizing to a fixed size
img = img / 255.0 # Normalizing the pixel values

# Detecting block copying/pasting
patch_size = 8 # The size of each patch
threshold = 0.1 # The entropy difference threshold
flags = np.zeros_like(img) # A matrix to store the flags
for i in range(0, img.shape[0], patch_size): # Looping over the rows
    for j in range(0, img.shape[1], patch_size): # Looping over the columns
        patch = img[i:i+patch_size, j:j+patch_size] # Extracting a patch
        hist = np.histogram(patch, bins=256)[0] # Computing the histogram
        entropy = -np.sum(hist * np.log2(hist + 1e-9)) # Computing the entropy
        if i > 0: # Checking the upper patch
            upper_patch = img[i-patch_size:i, j:j+patch_size]
            upper_hist = np.histogram(upper_patch, bins=256)[0]
            upper_entropy = -np.sum(upper_hist * np.log2(upper_hist + 1e-9))
            if abs(entropy - upper_entropy) < threshold: # Comparing the entropies
                flags[i:i+patch_size, j:j+patch_size] = 1 # Raising a flag
                flags[i-patch_size:i, j:j+patch_size] = 1 
        if j > 0: # Checking the left patch
            left_patch = img[i:i+patch_size, j-patch_size:j]
            left_hist = np.histogram(left_patch, bins=256)[0]
            left_entropy = -np.sum(left_hist * np.log2(left_hist + 1e-9))
            if abs(entropy - left_entropy) < threshold:
                flags[i:i+patch_size, j:j+patch_size] = 1 
                flags[i:i+patch_size, j-patch_size:j] = 1 

# Analyzing metadata
im = Image.open("input.jpg") # Opening the image with PIL library
metadata = im.getexif() # Extracting the metadata
for tag_id in metadata: # Looping over the metadata tags
    tag_name = TAGS.get(tag_id, tag_id) # Getting the tag name
    value = metadata.get(tag_id) # Getting the tag value
    print(f"{tag_name}: {value}") # Printing the tag name and value

# Applying spectral analysis
f_img = fft.fft2(img) # Applying Fourier Transform to the image
f_img_shifted = fft.fftshift(f_img) # Shifting the zero-frequency component to the center of spectrum
magnitude_spectrum = np.log(np.abs(f_img_shifted)) # Computing the magnitude spectrum

# Inspecting pixel ordering
ref_img = cv2.imread("reference.jpg") # Reading a reference image from file
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) # Converting to grayscale
ref_img = cv2.resize(ref_img, (512, 512)) # Resizing to match the input image size
ref_img = ref_img / 255.0 # Normalizing the pixel values

img_dct = cv2.dct(img) # Applying DCT to the input image
ref_dct = cv2.dct(ref_img) # Applying DCT to the reference image

diff_dct = np.abs(img_dct - ref_dct) # Computing the absolute difference between DCT coefficients

# Identifying compression artifacts
bit_depth = img.dtype.itemsize * 8 # Computing the bit depth of the image
compression_ratio = os.path.getsize("input.jpg") / (img.shape[0] * img.shape[1] * bit_depth / 8) # Computing the compression ratio of the image
print(f"Bit depth: {bit_depth}")
print(f"Compression ratio: {compression_ratio}")

# Investigating file format conversions
header = open("input.jpg", "rb").read(2) # Reading the first two bytes of the file
if header == b"\xff\xd8": # Checking if the header matches JPEG format
    print("The file format is JPEG")
else:
    print("The file format is not JPEG")

footer = open("input.jpg", "rb").read()[-2:] # Reading the last two bytes of the file
if footer == b"\xff\xd9": # Checking if the footer matches JPEG format
    print("The file format is JPEG")
else:
    print("The file format is not JPEG")

# Plotting and showing the results
plt.figure(figsize=(12, 12)) # Creating a figure with a large size
plt.subplot(2, 3, 1) # Creating a subplot in the first position
plt.imshow(img, cmap="gray") # Showing the input image
plt.title("Input image") # Setting the title of the subplot
plt.axis("off") # Turning off the axis

plt.subplot(2, 3, 2) # Creating a subplot in the second position
plt.imshow(flags, cmap="gray") # Showing the flags matrix
plt.title("Block copying/pasting detection") # Setting the title of the subplot
plt.axis("off") # Turning off the axis

plt.subplot(2, 3, 3) # Creating a subplot in the third position
plt.imshow(magnitude_spectrum, cmap="gray") # Showing the magnitude spectrum
plt.title("Spectral analysis") # Setting the title of the subplot
plt.axis("off") # Turning off the axis

plt.subplot(2, 3, 4) # Creating a subplot in the fourth position
plt.imshow(ref_img, cmap="gray") # Showing the reference image
plt.title("Reference image") # Setting the title of the subplot
plt.axis("off") # Turning off the axis

plt.subplot(2, 3, 5) # Creating a subplot in the fifth position
plt.imshow(diff_dct, cmap="gray") # Showing the difference between DCT coefficients
plt.title("Pixel ordering inspection") # Setting the title of the subplot
plt.axis("off") # Turning off the axis

plt.show() # Showing the figure

