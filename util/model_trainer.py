"""
ModelTrainer Module for Cross-lingual Sentiment Classification

This module provides a comprehensive training framework for BERT-based sentiment 
classification models with support for cross-lingual evaluation, mixed precision 
training, and automatic checkpointing.

Features:
- Mixed precision training for faster GPU training
- Cross-lingual evaluation on multiple datasets
- Automatic model checkpointing
- Training history tracking and persistence
- Learning rate scheduling support
- Progress bars with tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import os
import json
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


class ModelTrainer:
    """
    A comprehensive trainer for BERT sentiment classification models with cross-lingual evaluation.
    
    This trainer supports mixed precision training, automatic checkpointing, learning rate scheduling,
    and tracks performance across English and target language datasets.
    
    Attributes:
        model (nn.Module): The neural network model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to run training on (cuda/cpu)
        epochs (int): Number of training epochs
        english_train_loader (DataLoader): DataLoader for English training data
        english_val_loader (DataLoader): DataLoader for English validation data
        target_val_loader (DataLoader): DataLoader for target language validation data
        target_test_loader (DataLoader): DataLoader for target language test data
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Optional learning rate scheduler
        criterion (nn.CrossEntropyLoss): Loss function for training
        save_dir (str): Directory to save checkpoints and history
        use_mixed_precision (bool): Whether to use mixed precision training
        scaler (GradScaler): Gradient scaler for mixed precision training
    """
    
    def __init__(self, model, optimizer, device, epochs, english_train_loader, english_val_loader, 
                 target_val_loader, target_test_loader, lr_scheduler=None, save_dir="checkpoints", 
                 use_mixed_precision=True):
        """
        Initialize the ModelTrainer.
        
        Args:
            model (nn.Module): The neural network model to train
            optimizer (torch.optim.Optimizer): Optimizer for training (e.g., AdamW)
            device (torch.device): Device to run training on (cuda/cpu)
            epochs (int): Number of training epochs
            english_train_loader (DataLoader): DataLoader for English training data
            english_val_loader (DataLoader): DataLoader for English validation data
            target_val_loader (DataLoader): DataLoader for target language validation data
            target_test_loader (DataLoader): DataLoader for target language test data
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            save_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints"
            use_mixed_precision (bool, optional): Use mixed precision training. Defaults to True
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.english_train_loader = english_train_loader
        self.english_val_loader = english_val_loader
        self.target_val_loader = target_val_loader
        self.target_test_loader = target_test_loader
        self.lr_scheduler = lr_scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        
        # Mixed precision setup - only use on CUDA devices
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Performs training with the following features:
        - Mixed precision training (if enabled and on CUDA)
        - Progress bars for training progress
        - Validation on both English and target language datasets
        - Learning rate scheduling (if scheduler provided)
        - Automatic model checkpointing after each epoch
        - Training history tracking and saving
        
        Returns:
            dict: Training history containing losses, accuracies, F1-scores, and learning rates
                - train_losses (list): Training loss for each epoch
                - train_accuracies (list): Training accuracy for each epoch
                - train_f1_scores (list): Training macro F1-score for each epoch
                - english_val_losses (list): English validation loss for each epoch
                - english_val_accuracies (list): English validation accuracy for each epoch
                - english_val_f1_scores (list): English validation macro F1-score for each epoch
                - target_val_losses (list): Target validation loss for each epoch
                - target_val_accuracies (list): Target validation accuracy for each epoch
                - target_val_f1_scores (list): Target validation macro F1-score for each epoch
                - learning_rates (list): Learning rate for each epoch
        """
        self.model.train()
        
        # Initialize history tracking dictionary
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'train_f1_scores': [],
            'english_val_losses': [],
            'english_val_accuracies': [],
            'english_val_f1_scores': [],
            'target_val_losses': [],
            'target_val_accuracies': [],
            'target_val_f1_scores': [],
            'learning_rates': [],
            # 'val_roc_auc': [], # add ROC auc
            # 'target_val_roc_auc':[]
        }
        
        # Training loop with epoch progress bar
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            total_loss = 0
            train_predictions = []  # Store predictions for accuracy calculation
            train_labels = []       # Store true labels for accuracy calculation
            
            # Training batches with progress bar
            for batch in tqdm(self.english_train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                # Move batch to device
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    # Use autocast for forward pass
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        loss = self.criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward and backward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Collect training predictions for accuracy calculation
                preds = torch.argmax(outputs, dim=1)
                train_predictions.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # Step learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Calculate epoch metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / len(self.english_train_loader)
            train_acc = accuracy_score(train_labels, train_predictions)
            train_f1 = f1_score(train_labels, train_predictions, average='macro')
            
            # Evaluate on validation sets
            eng_val_loss, eng_val_acc, eng_val_f1, eng_probs, eng_labels = self.evaluate(self.english_val_loader, "English Val")
            target_val_loss, target_val_acc, target_val_f1, target_probs, target_labels = self.evaluate(self.target_val_loader, "Target Val")
            
            # No Need To calculate ROC auc in Training
            # # calculate English validation ROC auc
            # _, _, eng_roc_auc = self.compute_roc(eng_probs, eng_labels)
            # history['val_roc_auc'].append(eng_roc_auc)

            # # calculate target validation ROC auc
            # _, _, tgt_roc_auc = self.compute_roc(target_probs, target_labels)
            # history.setdefault('target_val_roc_auc', []).append(tgt_roc_auc)

            # Store metrics in history
            history['train_losses'].append(avg_loss)
            history['train_accuracies'].append(train_acc)
            history['train_f1_scores'].append(train_f1)
            history['english_val_losses'].append(eng_val_loss)
            history['english_val_accuracies'].append(eng_val_acc)
            history['english_val_f1_scores'].append(eng_val_f1)
            history['target_val_losses'].append(target_val_loss)
            history['target_val_accuracies'].append(target_val_acc)
            history['target_val_f1_scores'].append(target_val_f1)
            history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | LR: {current_lr:.2e}")
            print(f"  English Val - Loss: {eng_val_loss:.4f}, Acc: {eng_val_acc:.4f}, F1: {eng_val_f1:.4f}")
            print(f"  Target Val  - Loss: {target_val_loss:.4f}, Acc: {target_val_acc:.4f}, F1: {target_val_f1:.4f}")
            
            # Save model checkpoint after each epoch
            self.save_model(f"model_epoch_{epoch+1}.pth")
        
        # Save complete training history to file
        self.save_history(history)
        
        return history

    def evaluate(self, data_loader, dataset_name=""):
        """
        Evaluate the model on a given dataset.
        
        Args:
            data_loader (DataLoader): DataLoader containing the evaluation data
            dataset_name (str, optional): Name of the dataset for progress bar display
            
        Returns:
            tuple: (average_loss, accuracy, f1_score, probabilities, true_labels)
                - average_loss (float): Average loss across all batches
                - accuracy (float): Classification accuracy (0-1)
                - f1_score (float): Macro-averaged F1-score (0-1)
                - probabilities (np.array): Predicted class probabilities
                - true_labels (np.array): True labels
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        predictions = []
        true_labels = []
        all_probs = []
        
        # Evaluation loop without gradient computation
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}", leave=False):
                # Move batch to device
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                
                # Accumulate loss and predictions
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        # Return to training mode
        self.model.train()
        return avg_loss, accuracy, f1, np.array(all_probs), np.array(true_labels)

    def compute_roc(self, probs, true_labels):
        """calculate multiclass ROC and AUC"""
        true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2])
        n_classes = true_labels_bin.shape[1]
        # ROC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        return fpr, tpr, roc_auc

    def get_roc_data(self, data_loader):
        """get data to plot ROC curve"""
        _, _, _, probs, labels = self.evaluate(data_loader)
        return self.compute_roc(probs, labels)

    def test(self):
        """
        Evaluate the model on the target language test set.
        
        Returns:
            tuple: (test_loss, test_accuracy, test_f1_score)
                - test_loss (float): Average test loss
                - test_accuracy (float): Test accuracy (0-1)
                - test_f1_score (float): Macro-averaged F1-score (0-1)
        """
        test_loss, test_acc, test_f1, _, _ = self.evaluate(self.target_test_loader, "Target Test")
        print(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        return test_loss, test_acc, test_f1

    def save_model(self, filename):
        """
        Save the model state dictionary to a file.
        
        Args:
            filename (str): Name of the file to save the model to
        """
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        
    def save_history(self, history):
        """
        Save training history to a JSON file.
        
        Args:
            history (dict): Training history dictionary containing metrics
        """
        filepath = os.path.join(self.save_dir, "training_history.json")
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {filepath}")
        
    def load_model(self, filename):
        """
        Load a model state dictionary from a file.
        
        Args:
            filename (str): Name of the file to load the model from
        """
        filepath = os.path.join(self.save_dir, filename)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        
    def load_history(self, filename="training_history.json"):
        """
        Load training history from a JSON file.
        
        Args:
            filename (str, optional): Name of the history file. Defaults to "training_history.json"
            
        Returns:
            dict: Training history dictionary containing metrics
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'r') as f:
            history = json.load(f)
        return history