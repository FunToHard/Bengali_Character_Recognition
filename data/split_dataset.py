import os
import random
import shutil
from tqdm import tqdm

def create_train_val_split(root_dir, train_ratio=0.8):
    # Get train and val directories
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    # Create validation directory if it doesn't exist
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Get all character folders
    char_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    
    print(f"Found {len(char_folders)} character classes")
    total_moved = 0
    
    # Process each character folder
    for char in tqdm(char_folders, desc="Processing characters"):
        # Create corresponding folder in validation directory
        train_char_dir = os.path.join(train_dir, char)
        val_char_dir = os.path.join(val_dir, char)
        
        if not os.path.exists(val_char_dir):
            os.makedirs(val_char_dir)
        
        # Get all images in the character folder
        images = [f for f in os.listdir(train_char_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Calculate number of images for validation
        num_val = int(len(images) * (1 - train_ratio))
        
        # Randomly select images for validation
        val_images = random.sample(images, num_val)
        
        # Move selected images to validation folder
        for img in val_images:
            src = os.path.join(train_char_dir, img)
            dst = os.path.join(val_char_dir, img)
            shutil.move(src, dst)
            total_moved += 1
    
    return total_moved

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir  # The script is already in the data directory
    
    print("This script will split the dataset into training and validation sets")
    print("80% of images will remain in training, 20% will be moved to validation")
    print("Original folder structure will be maintained")
    response = input("Do you want to continue? (yes/no): ")
    
    if response.lower() == 'yes':
        # Set random seed for reproducibility
        random.seed(42)
        
        # Perform the split
        total_moved = create_train_val_split(root_dir)
        
        print(f"\nDataset split complete!")
        print(f"Moved {total_moved} images to validation set")
        
        # Print final statistics
        train_dir = os.path.join(root_dir, 'train')
        val_dir = os.path.join(root_dir, 'val')
        
        total_train = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                          for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        total_val = sum(len([f for f in os.listdir(os.path.join(val_dir, d)) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
        
        print("\nFinal Dataset Statistics:")
        print(f"Training images: {total_train}")
        print(f"Validation images: {total_val}")
        print(f"Total dataset size: {total_train + total_val}")
        print(f"Validation ratio: {total_val/(total_train + total_val):.2%}")
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    main()
