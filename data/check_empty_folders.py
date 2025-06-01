import os

def check_and_remove_empty_folders(directory):
    empty_folders = []
    all_folders = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            # Check if the folder is empty (no files and no subdirectories)
            if not os.listdir(folder_path):
                empty_folders.append(folder_path)
            all_folders.append(folder_path)
    
    if empty_folders:
        print(f"\nFound {len(empty_folders)} empty folders out of {len(all_folders)} total folders:")
        for folder in empty_folders:
            print(f"- {os.path.basename(folder)}")
        
        response = input("\nDo you want to remove these empty folders? (yes/no): ")
        if response.lower() == 'yes':
            for folder in empty_folders:
                try:
                    os.rmdir(folder)
                    print(f"Removed: {os.path.basename(folder)}")
                except Exception as e:
                    print(f"Error removing {os.path.basename(folder)}: {str(e)}")
            print("\nEmpty folders have been removed.")
        else:
            print("\nNo folders were removed.")
    else:
        print("\nNo empty folders found!")
    
    # Print statistics about remaining folders
    remaining_folders = [f for f in all_folders if os.path.exists(f)]
    if remaining_folders:
        print(f"\nCharacter distribution in remaining folders:")
        for folder in sorted(remaining_folders):
            num_files = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if num_files > 0:
                print(f"{os.path.basename(folder)}: {num_files} images")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "train")
    val_dir = os.path.join(script_dir, "val")
    
    print("Checking training directory...")
    check_and_remove_empty_folders(train_dir)
    
    print("\nChecking validation directory...")
    check_and_remove_empty_folders(val_dir)
    
    # Print overall dataset statistics
    print("\nOverall Dataset Statistics:")
    total_train = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                      for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    total_val = sum(len([f for f in os.listdir(os.path.join(val_dir, d)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Total dataset size: {total_train + total_val} images")

if __name__ == '__main__':
    main()
