import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import get_model
from utils import visualize_predictions, plot_confusion_matrix
import os
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, val_loader, device):
    """
    Evaluate model performance on the validation set
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def main():
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU')

    # Hyperparameters
    BATCH_SIZE = 128
    NUM_CLASSES = 62

    # Get validation data loader
    _, val_loader = get_data_loaders('data', BATCH_SIZE)

    # Load model
    model = get_model(NUM_CLASSES, device)
    checkpoint = torch.load('models/checkpoint_epoch_49.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print("\nEvaluating model performance...")
    
    # Get overall accuracy and predictions
    accuracy, all_preds, all_labels = evaluate_model(model, val_loader, device)
    print(f"\nOverall Validation Accuracy: {accuracy:.2f}%")
    
    # Generate and save classification report
    target_names = [str(i) for i in range(NUM_CLASSES)]  # You may want to map these to actual Bengali characters
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save the report to a file
    with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Overall Validation Accuracy: {accuracy:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(model, val_loader, device, NUM_CLASSES)
    
    # Visualize sample predictions
    print("\nGenerating sample predictions visualization...")
    visualize_predictions(model, val_loader, device)
    
    print("\nEvaluation complete. Files generated:")
    print("- evaluation_report.txt (Detailed metrics)")
    print("- confusion_matrix.png (Confusion matrix visualization)")
    print("- sample_predictions.png (Sample predictions visualization)")

if __name__ == '__main__':
    main()
