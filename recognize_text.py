import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model import get_model
import os
import cv2

def load_model(model_path, device):
    """
    Load the trained model from checkpoint
    """
    num_classes = 62  # Number of Bengali characters
    model = get_model(num_classes, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    """
    Preprocess an image for model inference
    """
    if isinstance(image, str):
        image = Image.open(image).convert('L')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('L')
    
    # Apply same transformations as validation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_bengali_char_map():
    """
    Create a mapping from class indices to Bengali characters using actual Bengali characters
    from the dataset folders
    """
    # Get character list from train directory
    data_dir = os.path.join('data', 'train')
    chars = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Create mapping with actual Bengali characters
    char_map = {i: char for i, char in enumerate(chars)}
    
    # Print the available characters for reference
    print("\nAvailable Bengali characters:")
    for i, char in char_map.items():
        print(f"{i}: {char}", end='  ')
        if (i + 1) % 8 == 0:  # Print 8 characters per line
            print()
    print("\n")
    
    return char_map

def segment_characters(image_path):
    """
    Segment an image into individual characters
    Returns a list of character images
    """
    # Read image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(enumerate(bounding_boxes), key=lambda x: x[1][0])  # Sort by x coordinate
    
    character_images = []
    original_positions = []  # To keep track of character positions
    min_size = 10  # Minimum size to be considered a character
    
    for idx, (i, (x, y, w, h)) in enumerate(sorted_boxes):
        if w > min_size and h > min_size:  # Filter out noise
            # Add padding around the character
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            # Extract character image
            char_image = gray[y_start:y_end, x_start:x_end]
            
            # Ensure the image is not empty
            if char_image.size > 0:
                character_images.append(char_image)
                original_positions.append((x, idx))
    
    # Sort characters by their original x-position
    sorted_chars = [img for _, img in sorted(zip([pos[0] for pos in original_positions], character_images))]
    return sorted_chars

def recognize_character(model, image_tensor, device):
    """
    Recognize a single Bengali character from a preprocessed image tensor
    """
    image_tensor = image_tensor.to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)
    
    return pred_class.item(), pred_prob.item()

def recognize_text(model, image_path, device, char_map):
    """
    Recognize multiple Bengali characters from an image
    Returns list of (character, confidence) tuples
    """
    # Segment the image into individual characters
    character_images = segment_characters(image_path)
    
    results = []
    for char_image in character_images:
        # Preprocess the character image
        image_tensor = preprocess_image(char_image)
        
        # Recognize the character
        pred_class, confidence = recognize_character(model, image_tensor, device)
        bengali_char = char_map[pred_class]
        results.append((bengali_char, confidence))
    
    return results

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model_path = 'models/checkpoint_epoch_49.pth'
    model = load_model(model_path, device)
    
    # Get Bengali character mapping
    char_map = get_bengali_char_map()
    
    # Example usage
    while True:
        image_path = input("\nEnter the path to image file (or 'q' to quit): ")
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("Error: File not found!")
            continue
            
        try:
            # Recognize text
            results = recognize_text(model, image_path, device, char_map)
            
            print("\nRecognized text:")
            text = ""
            print("\nDetailed results:")
            for char, confidence in results:
                text += char
                print(f"Character: {char}, Confidence: {confidence*100:.2f}%")
            
            print(f"\nComplete text: {text}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
