import os
import shutil
import random

def parse_label_file(label_file):
    # Parse the label file and return a dictionary {image_name: label}
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                image_path, label = parts
                image_name = os.path.basename(image_path)  # Extract just the image filename
                label_dict[image_name] = label
    return label_dict

def write_to_file(file_path, image_paths, labels, destination_dir):
    # Write image paths and corresponding labels to a text file
    with open(file_path, 'w') as f:
        for image in image_paths:
            label = labels.get(image, "UNKNOWN")  # Use "UNKNOWN" if no label found
            image_new_path = os.path.join(destination_dir, image)  # New path in the destination folder
            f.write(f"{image_new_path} {label}\n")

def split_dataset(source_dir, label_file, train_dir, valid_dir, test_dir, train_txt, valid_txt, test_txt, train_ratio=0.7, valid_ratio=0.2):
    # Create directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Parse the label file
    labels = parse_label_file(label_file)

    # List all .png files in the source directory
    images = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    random.shuffle(images)  # Shuffle the images for randomness

    total_images = len(images)
    
    # Calculate sizes based on ratios
    train_size = int(total_images * train_ratio)
    valid_size = int(total_images * valid_ratio)

    # Determine the split indices
    train_images = images[:train_size]
    valid_images = images[train_size:train_size + valid_size]
    test_images = images[train_size + valid_size:]  # Remaining images for test

    # Copy images to respective directories and write to .txt files
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(train_dir, image))
    for image in valid_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(valid_dir, image))
    for image in test_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(test_dir, image))

    # Write to train.txt, valid.txt, and test.txt
    write_to_file(train_txt, train_images, labels, train_dir)
    write_to_file(valid_txt, valid_images, labels, valid_dir)
    write_to_file(test_txt, test_images, labels, test_dir)

# Example usage:
split_dataset(
    source_dir='data_words/data', 
    label_file='data_words/gt.txt',
    train_dir='data_lmdb/icdar/train', 
    valid_dir='data_lmdb/icdar/valid', 
    test_dir='data_lmdb/icdar/test', 
    train_txt='data_lmdb/icdar/train.txt', 
    valid_txt='data_lmdb/icdar/valid.txt', 
    test_txt='data_lmdb/icdar/test.txt'
)
