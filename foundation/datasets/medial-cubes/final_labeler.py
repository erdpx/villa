import os
import argparse
from tifffile import imread, imwrite
import numpy as np
from tools import detect_ridges
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def process_tiff(image):
    mask = (image > 0).astype(np.float32)
    binary_inverted = 1 - mask
    # Compute the Euclidean Distance Transform
    edt = distance_transform_edt(binary_inverted)
    expanded_structure = edt <= 2
    ridges = detect_ridges(expanded_structure.astype(np.float32))
    ridges = np.maximum(ridges, mask)
    bin_edges = np.histogram_bin_edges(ridges, bins=5)
    binned_data = np.digitize(ridges, bin_edges[1:]).astype(np.uint8) # we can also make them just binary!
    return binned_data

def process_folder(source_dir, dest_dir, pattern):
    """
    Traverse the source folder, process TIFF files matching the pattern,
    and save them to the destination folder while preserving the subfolder structure.
    """
    for root, dirs, files in tqdm(os.walk(source_dir)):
        for file in files:
            if file.endswith(pattern + ".tif"):  # Match files ending with the pattern
                # Full path to the file
                file_path = os.path.join(root, file)
                
                # Relative path for maintaining folder structure
                relative_path = os.path.relpath(root, source_dir)
                
                # Read the TIFF file
                image = imread(file_path)
                
                # Process the image
                processed_image = process_tiff(image)
                
                # Modify the filename: remove the pattern from the filename
                new_filename = file.replace(f"{pattern}", "")
                
                # Destination path
                dest_folder = os.path.join(dest_dir, relative_path)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, new_filename)
                '''
                copy_image = processed_image.copy()

                if np.all(processed_image[0,:,:] == 0):
                    copy_image[0,:,:] = 6
                
                if np.all(processed_image[-1,:,:] == 0):
                    copy_image[-1,:,:] = 6

                if np.all(processed_image[:,0,:] == 0):
                    copy_image[:,0,:] = 6
                
                if np.all(processed_image[:,-1,:] == 0):
                    copy_image[:,-1,:] = 6

                
                if np.all(processed_image[:,:,0] == 0):
                    copy_image[:,:,0] = 6
                
                if np.all(processed_image[:,:,-1] == 0):
                    copy_image[:,:,-1] = 6

                # Save the processed image'''
                imwrite(dest_path, processed_image)
                
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Process TIFF files and save to a new directory.")
    parser.add_argument("source", type=str, help="Path to the source directory containing the TIFF files.")
    parser.add_argument("destination", type=str, help="Path to the destination directory where files will be saved.")
    parser.add_argument("--pattern", type=str, default="_pattern", help="Pattern to match and remove from TIFF file names (default: '_pattern').")

    args = parser.parse_args()
    
    # Call the processing function
    process_folder(args.source, args.destination, args.pattern)

if __name__ == "__main__":
    main()

# source: harmonized-cubes, destination: some new folder, pattern _thin_clean
