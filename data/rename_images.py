import os
import shutil

def rename_images(directory):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        # Get the Bengali character (folder name)
        bengali_char = os.path.basename(root)
        
        # Counter for images in this character folder
        counter = 1
        
        # Only process image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for old_name in sorted(image_files):
            # Create new name: bengali_char_number.jpg
            # Example: if folder is 'অ', images will be অ_0001.jpg, অ_0002.jpg, etc.
            extension = os.path.splitext(old_name)[1]
            new_name = f"{bengali_char}_{counter:04d}{extension}"
            
            old_path = os.path.join(root, old_name)
            new_path = os.path.join(root, new_name)
            
            try:
                # Rename the file
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed '{old_name}' to '{new_name}' in {bengali_char}/")
                else:
                    print(f"Skipped '{old_name}' as '{new_name}' already exists in {bengali_char}/")
                counter += 1
            except Exception as e:
                print(f"Error renaming '{old_name}' in {bengali_char}/: {str(e)}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "..", "train")
    val_dir = os.path.join(script_dir, "..", "val")
    
    print("This script will rename all image files in the train and val directories.")
    print("New format will be: BengaliChar_0001.jpg")
    print("Make sure you have a backup of your data before proceeding.")
    response = input("Do you want to continue? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\nProcessing training directory...")
        rename_images(train_dir)
        print("\nProcessing validation directory...")
        rename_images(val_dir)
        print("\nRenaming complete!")
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    main()
