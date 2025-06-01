import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import get_model
import os
import math
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, filename)

def load_latest_checkpoint(model, optimizer, scheduler, device):
    checkpoint_files = sorted([f for f in os.listdir('models') if f.startswith('checkpoint_epoch_')])
    if not checkpoint_files:
        return None, 0
    
    latest_checkpoint = os.path.join('models', checkpoint_files[-1])
    print(f'Loading checkpoint: {latest_checkpoint}')
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_loss']

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001, start_epoch=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
            
            # Update learning rate
            scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
        
        val_loss = running_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join('models', f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_accuracy, checkpoint_path)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # Save final plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join('models', f'training_history_{timestamp}.png'))
    plt.close()

def main():
    # Check CUDA availability and print detailed GPU info
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU')
    
    print(f'PyTorch CUDA version: {torch.version.cuda}')
    
    # Hyperparameters
    BATCH_SIZE = 128  # Increased batch size
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 62  # Based on the number of Bengali character folders
    
    # Get data loaders
    data_dir = 'data'
    train_loader, val_loader = get_data_loaders(data_dir, BATCH_SIZE)
    
    # Initialize model
    model = get_model(NUM_CLASSES, device)
      # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Load latest checkpoint if exists
    start_epoch, best_val_loss = load_latest_checkpoint(model, optimizer, scheduler, device)
    if start_epoch > 0:
        print(f'Resuming training from epoch {start_epoch + 1}')
    
    # Train model
    train_model(model, train_loader, val_loader, NUM_EPOCHS, device, LEARNING_RATE, start_epoch=start_epoch)

if __name__ == '__main__':
    main()
