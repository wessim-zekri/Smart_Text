import os

def count_images_in_directory(directory):
    """
    Count the number of image files in a given directory.
    
    Args:
    - directory (str): Path to the directory to count images in.

    Returns:
    - int: Count of image files in the directory.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_count = 0
    
    # Check if the directory exists
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            # Check if the file is an image based on the extension
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_count += 1
    else:
        print(f"Directory '{directory}' does not exist.")
    
    return image_count

# Specify the paths to your dataset directories
train_dir = 'data_lmdb/icdar/train'  # Update this path accordingly
val_dir = 'data_lmdb/icdar/valid'     # Update this path accordingly
test_dir = 'data_lmdb/icdar/test'      # Update this path accordingly

# Count images
train_count = count_images_in_directory(train_dir)
val_count = count_images_in_directory(val_dir)
test_count = count_images_in_directory(test_dir)

# Print the results
print(f"Number of training images: {train_count}")
print(f"Number of validation images: {val_count}")
print(f"Number of test images: {test_count}")
