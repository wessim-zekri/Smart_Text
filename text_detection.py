import cv2
import numpy as np
import torch
from craft import CRAFT  # Assuming you have the CRAFT model code
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_craft_model(weights_path):
    craft_net = CRAFT()  # Initialize the CRAFT model
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Print the keys in the checkpoint to inspect its structure
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if the state_dict is directly under the checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # If there is no 'state_dict', use the checkpoint directly
        state_dict = checkpoint
    
    # If model was trained using DataParallel, we may need to remove the 'module.' prefix
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        state_dict = new_state_dict

    # Load the state_dict into the model
    craft_net.load_state_dict(state_dict)
    
    return craft_net

def detect_text_boxes(image, craft_net, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False):
    """Detect text boxes in the given image using the CRAFT model."""
    # Pre-process the image
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # Normalize mean and variance
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    # Forward pass
    #with torch.no_grad():
    y = craft_net(x)  # y is the output from the model
    print(f"craft_net output: {y}")
    print(y[0].shape)  # or inspect the content of y

    # Debugging the output of the CRAFT model
    # If y is a single output, handle it accordingly
    if isinstance(y, (tuple, list)) and len(y) == 2:
        textmap, linkmap = y[0], y[1]
    else:
        textmap = y[0]  # Assuming y[0] is the textmap
        linkmap = None   # No linkmap available in this case

    print(f"textmap shape: {textmap.shape}")
    print(f"linkmap shape: {linkmap.shape}")

    # Ensure linkmap is a NumPy array and properly formatted before passing to getDetBoxes
    if isinstance(linkmap, torch.Tensor):
        linkmap = linkmap.cpu().detach().numpy()  # Convert to NumPy array if needed

    if not isinstance(linkmap, np.ndarray):
        raise ValueError(f"Expected linkmap to be a NumPy array, but got {type(linkmap)}")

    # Get text boxes
    boxes, polys = getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly)
    
    # Adjust coordinates according to the original image
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Convert boxes to integer coordinates
    boxes = [list(map(int, box.ravel())) for box in boxes]  # Flatten box points
    return boxes
