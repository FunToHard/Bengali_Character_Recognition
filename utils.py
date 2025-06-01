import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from model import get_model
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    Visualize model predictions on sample images
    """
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert tensor to numpy for visualization
            img = images[0].cpu().numpy().squeeze()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'True: {labels[0].item()}\nPred: {predicted[0].item()}')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

def plot_confusion_matrix(model, test_loader, device, num_classes):
    """
    Plot confusion matrix for model predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

def predict_single_image(model, image_path, device):
    """
    Make prediction on a single image
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()
