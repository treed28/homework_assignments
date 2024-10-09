import numpy as np
import os
from astropy.io import fits


def load_fits_images(directory):
    """
    Load FITS files that start with "Bias" and end with ".fit" from the specified directory.
    
    Parameters:
    directory: str
        The path to the directory containing FITS files.
        
    Returns:
    bias_images: List of numpy arrays representing the loaded bias images.
    """
    bias_images = []
    
    # Loop through all files in the directory that match the "Bias*.fit" pattern
    for file_name in sorted(os.listdir(directory)):
        if file_name.startswith("Bias") and file_name.endswith(".fit"):  # Check for "Bias" prefix and ".fit" extension
            # Load the FITS file
            file_path = os.path.join(directory, file_name)
            try:
                with fits.open(file_path) as hdul:
                    bias_image = hdul[0].data  # Get the image data from the FITS file
                    if bias_image is not None:  # Ensure the data is not None
                        bias_images.append(np.array(bias_image,dtype=float))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    # Check if any images were loaded
    if len(bias_images) == 0:
        print(f"No valid FITS files were loaded from {directory}.")
    
    return bias_images

def readout_noise_from_bias_images(bias_images):
    """
    Estimate the readout noise in ADU from pairs of bias images.
    
    Parameters:
    bias_images: List of numpy arrays representing the loaded bias images.
    
    Returns:
    average_readout_noise: The average readout noise in ADU, or None if calculation fails.
    """
    readout_noises = []

    # Ensure we have an even number of bias images to form pairs
    num_images = len(bias_images)
    if num_images == 0:
        print("No bias images to process.")
        return None

    if num_images % 2 != 0:
        print("Warning: Odd number of bias images. The last image will be ignored.")
        bias_images = bias_images[:-1]

    # Loop over pairs of bias images
    for i in range(0, len(bias_images), 2):
        B1 = bias_images[i]
        B2 = bias_images[i + 1]
        
        # Compute the difference between the two bias images
        difference = B2 - B1
        
        # Compute the variance of the difference
        variance_diff = np.var(difference)
        
        # Calculate the readout noise for this pair of images
        sigma_read_ADU = np.sqrt(variance_diff / 2)
        
        # Append the result to the list of readout noises
        readout_noises.append(sigma_read_ADU)

    if len(readout_noises) == 0:
        print("No valid readout noise values were computed.")
        return None

    # Compute the average readout noise
    average_readout_noise = np.mean(readout_noises)
    
    return average_readout_noise


def load_fits_images_by_filter(directory, prefix, filter_type):
    """
    Load FITS files that start with the given prefix and end with a specific filter type (e.g., "B.fit", "V.fit", "R.fit").
    
    Parameters:
    directory: str
        The path to the directory containing FITS files.
    prefix: str
        The prefix of the files to load (e.g., "DomeFlat").
    filter_type: str
        The filter type suffix to select (e.g., "B.fit", "V.fit", "R.fit").
        
    Returns:
    images: List of numpy arrays representing the loaded images (converted to float).
    """
    images = []
    
    # Loop through all files in the directory and load files with the matching prefix and filter type suffix
    for file_name in sorted(os.listdir(directory)):
        if file_name.startswith(prefix) and file_name.endswith(filter_type):  # Check for matching prefix and filter type
            file_path = os.path.join(directory, file_name)
            try:
                with fits.open(file_path) as hdul:
                    image_data = hdul[0].data  # Get the image data from the FITS file
                    if image_data is not None:
                        images.append(np.array(image_data, dtype=float))  # Convert the image data to float
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    # Check if any images were loaded
    if len(images) == 0:
        print(f"No valid FITS files with prefix '{prefix}' and filter '{filter_type}' were loaded from {directory}.")
    
    return images


def calculate_gain_and_readout_noise(flat1, flat2, bias1, bias2):
    """
    Calculate the gain (g) in electrons/ADU and the readout noise in electrons.
    
    Parameters:
    flat1: numpy array, first flat field image (in ADU)
    flat2: numpy array, second flat field image (in ADU)
    bias1: numpy array, first bias image (in ADU)
    bias2: numpy array, second bias image (in ADU)
    
    Returns:
    gain: float, the gain in e-/ADU
    readout_noise_electrons: float, the readout noise in electrons
    """
    
    # Subtract bias from flat field images
    F1_minus_B1 = np.mean(flat1 - bias1)
    F2_minus_B2 = np.mean(flat2 - bias2)
    
    # Compute the variance of the difference between the flat field images
    sigma_F_diff = np.std(flat2 - flat1)
    
    # Compute the variance of the difference between the bias images
    sigma_B_diff = np.std(bias2 - bias1)
    
    # Calculate the gain using the formula
    gain = (F1_minus_B1 + F2_minus_B2) / (sigma_F_diff**2 - sigma_B_diff**2)
    
    # Estimate the readout noise in ADU
    sigma_read_ADU = np.sqrt(sigma_B_diff / 2)
    
    # Main process
    directory = '/home/idies/workspace/Temporary/treed28/scratch/20240903/'
    
    # Load the bias images (those starting with "Bias")
    bias_images = load_fits_images(directory) #, "Bias", ".fit")
    
    # Estimate the average readout noise
    average_noise = readout_noise_from_bias_images(bias_images)
    
    # Calculate the readout noise in electrons
    readout_noise_electrons = gain * average_noise
    
    return gain, readout_noise_electrons
