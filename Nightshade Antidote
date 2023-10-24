# Importing the required libraries
import cv2 # For image processing
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting and visualization
import scipy.fftpack as fft # For Fourier Transform
from PIL import Image # For metadata extraction
from collections import Counter # For histogram calculation
from sklearn.neighbors import NearestNeighbors # For KNN search
import exiftool # For metadata extraction

# Loading and preprocessing the input image
img = cv2.imread("input.jpg") # Reading the image from file
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting to grayscale
img = cv2.resize(img, (512, 512)) # Resizing to a fixed size
img = img / 255.0 # Normalizing the pixel values

# A function to detect block copying/pasting using KNN search
def detect_copy_move(img, patch_size=8, threshold=0.1):
    flags = np.zeros_like(img) # A matrix to store the flags
    dct_img = cv2.dct(img) # Applying DCT to the image
    patches = [] # A list to store the patches
    indices = [] # A list to store the patch indices
    for i in range(0, img.shape[0], patch_size): # Looping over the rows
        for j in range(0, img.shape[1], patch_size): # Looping over the columns
            patch = dct_img[i:i+patch_size, j:j+patch_size] # Extracting a patch
            patches.append(patch.flatten()) # Flattening and appending the patch to the list
            indices.append((i, j)) # Appending the patch index to the list
    
    patches = np.array(patches) # Converting the list of patches to a numpy array
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(patches) # Fitting a KNN model on the patches
    
    distances, neighbors = nbrs.kneighbors(patches) # Finding the distances and neighbors for each patch
    
    for i in range(len(patches)): # Looping over the patches
        if distances[i][1] < threshold: # If the distance to the second nearest neighbor is less than the threshold
            x1, y1 = indices[i] # Getting the index of the current patch
            x2, y2 = indices[neighbors[i][1]] # Getting the index of the second nearest neighbor patch
            flags[x1:x1+patch_size, y1:y1+patch_size] = 1 # Raising a flag for both patches
            flags[x2:x2+patch_size, y2:y2+patch_size] = 1 
    
    return flags

# A function to analyze metadata using ExifTool instead of PIL or OpenCV
def analyze_metadata(img):
    with exiftool.ExifTool() as et: # Creating an ExifTool object
        metadata = et.get_metadata("input.jpg") # Extracting the metadata from the image file using ExifTool
        for tag in metadata: # Looping over the metadata tags
            print(f"{tag}: {metadata[tag]}") # Printing the tag name and value

# A function to apply spectral analysis and split into magnitude and phase spectra and plot them separately
def spectral_analysis(img):
    f_img = fft.fft2(img) # Applying Fourier Transform to the image
    f_img_shifted = fft.fftshift(f_img) # Shifting the zero-frequency component to the center of spectrum
    
    magnitude_spectrum = np.log(np.abs(f_img_shifted)) # Computing the magnitude spectrum
    
    phase_spectrum = np.angle(f_img_shifted) # Computing the phase spectrum
    
    plt.figure(figsize=(10, 5)) # Creating a figure with a fixed size
    
    plt.subplot(121) # Creating a subplot for magnitude spectrum
    plt.imshow(magnitude_spectrum) # Plotting the magnitude spectrum
    plt.title("Magnitude Spectrum") # Adding a title to the plot
    
    plt.subplot(122) # Creating a subplot for phase spectrum
    plt.imshow(phase_spectrum) # Plotting the phase spectrum
    plt.title("Phase Spectrum") # Adding a title to the plot
    
    plt.show() # Showing both plots
    
    return magnitude_spectrum, phase_spectrum

# A function to inspect pixel ordering by comparing DCT coefficients with a reference image after alignment and normalization using correlation coefficient as a metric
def pixel_ordering_check(img):
    ref_img = cv2.imread("reference.jpg") # Reading a reference image from file
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) # Converting to grayscale
    
    img_dct = cv2.dct(img) # Applying DCT to the input image
    
    ref_dct = cv2.dct(ref_img) # Applying DCT to the reference image
    
    img_dct_normed = (img_dct - img_dct.mean()) / img_dct.std() # Normalizing the input DCT coefficients
    
    ref_dct_normed = (ref_dct - ref_dct.mean()) / ref_dct.std() # Normalizing the reference DCT coefficients
    
    corr_coeff = np.corrcoef(img_dct_normed.flatten(), ref_dct_normed.flatten())[0, 1] # Computing the correlation coefficient between the normalized DCT coefficients
    
    return corr_coeff

# A function to identify compression artifacts by checking JPEG quantization artifacts in the DCT coefficients using a threshold
def compression_artifacts_check(img):
    bit_depth = img.dtype.itemsize * 8 # Computing the bit depth of the image
    compression_ratio = os.path.getsize("input.jpg") / (img.shape[0] * img.shape[1] * bit_depth / 8) # Computing the compression ratio of the image
    print(f"Bit depth: {bit_depth}")
    print(f"Compression ratio: {compression_ratio}")
    
    dct_img = cv2.dct(img) # Applying DCT to the image
    
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61], # The JPEG quantization table for luminance
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68,109,103, 77],
                                   [24, 35, 55, 64, 81,104,113, 92],
                                   [49, 64, 78, 87,103,121,120,101],
                                   [72, 92, 95, 98,112,100,103,99]])
    
    quantization_artifacts = dct_img % quantization_table # Computing the remainder of dividing DCT coefficients by quantization table
    
    threshold = quantization_table * 0.1 # Setting a threshold as a percentage of the quantization table values
    
    flags = np.zeros_like(img) # A matrix to store the flags
    
    flags[quantization_artifacts < threshold] = 1 # Raising a flag for any DCT coefficient that is less than the threshold
    
    return flags

# A function to investigate file format conversions by checking the header and footer bytes
def file_format_check(img):
    header = open("input.jpg", "rb").read(2) # Reading the first two bytes of the file
    if header == b"\xff\xd8": # Checking if the header matches JPEG format
        print("The file format is JPEG")
    else:
        print("The file format is not JPEG")

    footer = open("input.jpg", "rb").read()[-2:] # Reading the last two bytes
    if footer == b"\xff\xd9": # Checking if the footer matches JPEG format
        print("The file format is JPEG")
    else:
        print("The file format is not JPEG")

# A function to output a report summarizing the forensic findings
def output_report(img):
    report = "" # An empty string to store the report
    
    report += "Image Forensics Report\n"
    report += "=====================\n\n"
    
    report += "Metadata Analysis\n"
    report += "-----------------\n"
    metadata = analyze_metadata(img) # Calling the metadata analysis function
    report += metadata + "\n" # Adding the metadata to the report
    
    report += "Copy-Move Forgery Detection\n"
    report += "---------------------------\n"
    flags = detect_copy_move(img) # Calling the copy-move detection function
    num_flags = np.sum(flags) # Counting the number of flags raised
    if num_flags > 0: # If there are any flags raised
        report += f"Detected {num_flags} regions that are likely copied and pasted.\n" # Reporting the number of regions
        plt.imshow(flags) # Plotting the flags on an image
        plt.title("Copy-Move Regions") # Adding a title to the plot
        plt.show() # Showing the plot
    else: # If there are no flags raised
        report += "No copy-move forgery detected.\n" # Reporting no forgery
    
    report += "Spectral Analysis\n"
    report += "-----------------\n"
    magnitude_spectrum, phase_spectrum = spectral_analysis(img) # Calling the spectral analysis function and getting both spectra
    plt.figure(figsize=(10, 5)) # Creating a figure with a fixed size
    
    plt.subplot(121) # Creating a subplot for magnitude spectrum
    plt.imshow(magnitude_spectrum) # Plotting the magnitude spectrum
    plt.title("Magnitude Spectrum") # Adding a title to the plot
    
    plt.subplot(122) # Creating a subplot for phase spectrum
    plt.imshow(phase_spectrum) # Plotting the phase spectrum
    plt.title("Phase Spectrum") # Adding a title to the plot
    
    plt.show() # Showing both plots
    
    # Checking for any anomalies in the spectra
    mag_mean = np.mean(magnitude_spectrum) # Computing the mean of the magnitude spectrum
    mag_std = np.std(magnitude_spectrum) # Computing the standard deviation of the magnitude spectrum
    mag_threshold = mag_mean + 3 * mag_std # Setting a threshold as three standard deviations above the mean
    mag_anomalies = np.where(magnitude_spectrum > mag_threshold) # Finding the indices where the magnitude spectrum exceeds the threshold
    if len(mag_anomalies[0]) > 0: # If there are any anomalies
        report += f"Detected {len(mag_anomalies[0])} anomalies in the magnitude spectrum.\n" # Reporting the number of anomalies
        report += "These anomalies may indicate hidden messages, periodic noise, or resampling in the image.\n" # Explaining the possible causes of anomalies
        plt.scatter(mag_anomalies[1], mag_anomalies[0], c='r', marker='x') # Plotting the anomalies on the magnitude spectrum
        plt.title("Magnitude Spectrum Anomalies") # Adding a title to the plot
        plt.show() # Showing the plot
    else: # If there are no anomalies
        report += "No anomalies detected in the magnitude spectrum.\n" # Reporting no anomalies
    
    phase_mean = np.mean(phase_spectrum) # Computing the mean of the phase spectrum
    phase_std = np.std(phase_spectrum) # Computing the standard deviation of the phase spectrum
    phase_threshold = phase_mean + 3 * phase_std # Setting a threshold as three standard deviations above the mean
    phase_anomalies = np.where(phase_spectrum > phase_threshold) # Finding the indices where the phase spectrum exceeds the threshold
    if len(phase_anomalies[0]) > 0: # If there are any anomalies
        report += f"Detected {len(phase_anomalies[0])} anomalies in the phase spectrum.\n" # Reporting the number of anomalies
        report += "These anomalies may indicate geometric transformations, such as rotation, scaling, or cropping in the image.\n" # Explaining the possible causes of anomalies
        plt.scatter(phase_anomalies[1], phase_anomalies[0], c='r', marker='x') # Plotting the anomalies on the phase spectrum
        plt.title("Phase Spectrum Anomalies") # Adding a title to the plot
        plt.show() # Showing the plot
    else: # If there are no anomalies
        report += "No anomalies detected in the phase spectrum.\n" # Reporting no anomalies
    
    return magnitude_spectrum, phase_spectrum
