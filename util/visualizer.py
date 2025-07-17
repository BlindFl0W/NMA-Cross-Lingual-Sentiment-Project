"""
Visualizer Module for Cross-lingual Sentiment Classification Training Results

This module provides comprehensive visualization functions for analyzing and plotting
training results from the ModelTrainer. It includes functions for plotting losses,
accuracies, learning rates, cross-lingual transfer performance, and creating
comprehensive training summaries.

Features:
- Training and validation loss plots
- Accuracy progression plots
- Learning rate schedule visualization
- Cross-lingual transfer gap analysis
- Comprehensive training dashboard
- Customizable plot styling
- Save plots to files
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd


class TrainingVisualizer:
    """
    A comprehensive visualizer for training results from cross-lingual sentiment classification.
    
    This class provides various plotting functions to analyze training progress,
    validation performance, and cross-lingual transfer capabilities.
    
    Attributes:
        style (str): Matplotlib/seaborn style for plots
        figsize (tuple): Default figure size for plots
        dpi (int): Resolution for saved plots
        color_palette (list): Color palette for consistent styling
    """
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8), dpi=300):
        """
        Initialize the TrainingVisualizer.
        
        Args:
            style (str): Matplotlib style to use for plots
            figsize (tuple): Default figure size (width, height)
            dpi (int): DPI for saved figures
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Define color palette for consistent styling
        self.color_palette = {
            'train': '#1f77b4',
            'english_val': '#ff7f0e', 
            'target_val': '#2ca02c',
            'lr': '#d62728',
            'transfer_gap': '#9467bd'
        }
    
    def plot_losses(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot training and validation losses over epochs.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Plot loss curves
        ax.plot(epochs, history['train_losses'], 
                color=self.color_palette['train'], marker='o', linewidth=2,
                label='Training Loss', markersize=4)
        ax.plot(epochs, history['english_val_losses'], 
                color=self.color_palette['english_val'], marker='s', linewidth=2,
                label='English Validation Loss', markersize=4)
        ax.plot(epochs, history['target_val_losses'], 
                color=self.color_palette['target_val'], marker='^', linewidth=2,
                label='Target Validation Loss', markersize=4)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add best loss annotation
        best_val_loss = min(history['english_val_losses'])
        best_epoch = history['english_val_losses'].index(best_val_loss) + 1
        ax.annotate(f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}',
                   xy=(best_epoch, best_val_loss), xytext=(best_epoch + 1, best_val_loss + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_accuracies(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot training and validation accuracies over epochs.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['train_accuracies']) + 1)
        
        # Plot accuracy curves
        ax.plot(epochs, history['train_accuracies'], 
                color=self.color_palette['train'], marker='o', linewidth=2,
                label='Training Accuracy', markersize=4)
        ax.plot(epochs, history['english_val_accuracies'], 
                color=self.color_palette['english_val'], marker='s', linewidth=2,
                label='English Validation Accuracy', markersize=4)
        ax.plot(epochs, history['target_val_accuracies'], 
                color=self.color_palette['target_val'], marker='^', linewidth=2,
                label='Target Validation Accuracy', markersize=4)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Accuracy Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add best accuracy annotation
        best_target_acc = max(history['target_val_accuracies'])
        best_epoch = history['target_val_accuracies'].index(best_target_acc) + 1
        ax.annotate(f'Best Target Acc: {best_target_acc:.4f}\nEpoch: {best_epoch}',
                   xy=(best_epoch, best_target_acc), xytext=(best_epoch + 1, best_target_acc - 0.05),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_learning_rate(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot learning rate schedule over epochs.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['learning_rates']) + 1)
        
        # Plot learning rate
        ax.plot(epochs, history['learning_rates'], 
                color=self.color_palette['lr'], marker='o', linewidth=2,
                label='Learning Rate', markersize=4)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Learning rate plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_transfer_gap(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot cross-lingual transfer gap (English accuracy - Target accuracy).
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['english_val_accuracies']) + 1)
        transfer_gap = np.array(history['english_val_accuracies']) - np.array(history['target_val_accuracies'])
        
        # Plot transfer gap
        ax.plot(epochs, transfer_gap, 
                color=self.color_palette['transfer_gap'], marker='d', linewidth=2,
                label='Cross-lingual Transfer Gap', markersize=4)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Transfer')
        
        # Color areas
        ax.fill_between(epochs, transfer_gap, 0, where=(transfer_gap >= 0), 
                       color='red', alpha=0.2, label='English Better')
        ax.fill_between(epochs, transfer_gap, 0, where=(transfer_gap < 0), 
                       color='green', alpha=0.2, label='Target Better')
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Gap (English - Target)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-lingual Transfer Performance Gap', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final gap annotation
        final_gap = transfer_gap[-1]
        ax.annotate(f'Final Gap: {final_gap:+.4f}',
                   xy=(len(epochs), final_gap), xytext=(len(epochs) - 1, final_gap + 0.02),
                   arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lavender', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Transfer gap plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_dashboard(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Create a comprehensive training dashboard with multiple subplots.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-lingual Sentiment Training Dashboard', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss plot (top-left)
        axes[0, 0].plot(epochs, history['train_losses'], color=self.color_palette['train'], 
                       marker='o', label='Training', linewidth=2, markersize=3)
        axes[0, 0].plot(epochs, history['english_val_losses'], color=self.color_palette['english_val'], 
                       marker='s', label='English Val', linewidth=2, markersize=3)
        axes[0, 0].plot(epochs, history['target_val_losses'], color=self.color_palette['target_val'], 
                       marker='^', label='Target Val', linewidth=2, markersize=3)
        axes[0, 0].set_title('Loss Progression', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot (top-right)
        axes[0, 1].plot(epochs, history['train_accuracies'], color=self.color_palette['train'], 
                       marker='o', label='Training', linewidth=2, markersize=3)
        axes[0, 1].plot(epochs, history['english_val_accuracies'], color=self.color_palette['english_val'], 
                       marker='s', label='English Val', linewidth=2, markersize=3)
        axes[0, 1].plot(epochs, history['target_val_accuracies'], color=self.color_palette['target_val'], 
                       marker='^', label='Target Val', linewidth=2, markersize=3)
        axes[0, 1].set_title('Accuracy Progression', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Learning rate plot (bottom-left)
        axes[1, 0].plot(epochs, history['learning_rates'], color=self.color_palette['lr'], 
                       marker='o', label='Learning Rate', linewidth=2, markersize=3)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Transfer gap plot (bottom-right)
        transfer_gap = np.array(history['english_val_accuracies']) - np.array(history['target_val_accuracies'])
        axes[1, 1].plot(epochs, transfer_gap, color=self.color_palette['transfer_gap'], 
                       marker='d', label='Transfer Gap', linewidth=2, markersize=3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, transfer_gap, 0, where=(transfer_gap >= 0), 
                               color='red', alpha=0.2)
        axes[1, 1].fill_between(epochs, transfer_gap, 0, where=(transfer_gap < 0), 
                               color='green', alpha=0.2)
        axes[1, 1].set_title('Cross-lingual Transfer Gap', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (English - Target)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Training dashboard saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_final_comparison(self, history: Dict, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Create a bar plot comparing final performance across datasets.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            save_path (str, optional): Path to save the plot
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Final accuracies
        datasets = ['Training', 'English Val', 'Target Val']
        final_accuracies = [
            history['train_accuracies'][-1],
            history['english_val_accuracies'][-1],
            history['target_val_accuracies'][-1]
        ]
        colors = [self.color_palette['train'], self.color_palette['english_val'], self.color_palette['target_val']]
        
        bars1 = ax1.bar(datasets, final_accuracies, color=colors, alpha=0.8)
        ax1.set_title('Final Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, final_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Final losses
        final_losses = [
            history['train_losses'][-1],
            history['english_val_losses'][-1],
            history['target_val_losses'][-1]
        ]
        
        bars2 = ax2.bar(datasets, final_losses, color=colors, alpha=0.8)
        ax2.set_title('Final Loss Comparison', fontweight='bold')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, loss in zip(bars2, final_losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Final comparison saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def save_all_plots(self, history: Dict, output_dir: str = "plots"):
        """
        Generate and save all visualization plots to a directory.
        
        Args:
            history (dict): Training history dictionary from ModelTrainer
            output_dir (str): Directory to save all plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating and saving all plots to {output_dir}/")
        
        # Generate all plots
        self.plot_losses(history, os.path.join(output_dir, "losses.png"), show=False)
        self.plot_accuracies(history, os.path.join(output_dir, "accuracies.png"), show=False)
        self.plot_learning_rate(history, os.path.join(output_dir, "learning_rate.png"), show=False)
        self.plot_transfer_gap(history, os.path.join(output_dir, "transfer_gap.png"), show=False)
        self.plot_training_dashboard(history, os.path.join(output_dir, "dashboard.png"), show=False)
        self.plot_final_comparison(history, os.path.join(output_dir, "final_comparison.png"), show=False)
        
        print(f"All plots saved successfully!")
    
    @staticmethod
    def load_history_from_file(filepath: str) -> Dict:
        """
        Load training history from a JSON file.
        
        Args:
            filepath (str): Path to the training history JSON file
            
        Returns:
            dict: Training history dictionary
        """
        with open(filepath, 'r') as f:
            history = json.load(f)
        return history


# Convenience functions for quick plotting
def quick_plot_losses(history_path: str, save_path: Optional[str] = None):
    """Quick function to plot losses from a history file."""
    visualizer = TrainingVisualizer()
    history = visualizer.load_history_from_file(history_path)
    visualizer.plot_losses(history, save_path)


def quick_plot_accuracies(history_path: str, save_path: Optional[str] = None):
    """Quick function to plot accuracies from a history file."""
    visualizer = TrainingVisualizer()
    history = visualizer.load_history_from_file(history_path)
    visualizer.plot_accuracies(history, save_path)


def quick_dashboard(history_path: str, save_path: Optional[str] = None):
    """Quick function to create training dashboard from a history file."""
    visualizer = TrainingVisualizer()
    history = visualizer.load_history_from_file(history_path)
    visualizer.plot_training_dashboard(history, save_path)