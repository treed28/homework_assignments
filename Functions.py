import numpy as np
import os
from astropy.io import fits



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
    
    # Calculate the readout noise in electrons
    readout_noise_electrons = gain * average_noise
    
    return gain, readout_noise_electrons