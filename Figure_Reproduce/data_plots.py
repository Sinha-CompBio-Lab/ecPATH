import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_plot(data_dict, metric='auc_scores', figsize=(12, 10), save_path="figure.png",feature_select=""):
    """
    Generate a radar plot comparing Syntentic and Real Data across cancer types.
    """

    # Define fixed order of all possible cancer types
    all_cancer_types = ["LGG", "LUSC", "CESC", "ESCA", "HNSC", "LUAD", "BRCA", "STAD", "GBM"]
    
    # Extract data for the specified metric
    methods = ['Synthetic', 'True']
    
    # Get available cancer types from the data
    available_cancer_types = set()
    for method in methods:
        available_cancer_types.update(data_dict[method].keys())


    # Filter cancer types to those present in the data while maintaining the fixed order
    cancer_types = [cancer for cancer in all_cancer_types if cancer in available_cancer_types]

    # Prepare data for plotting
    values = {
        'Synthetic': [],
        'True': []
    }
    
    # Get values for each method and cancer type, handling missing data
    for method in methods:
        for cancer in cancer_types:
            if cancer in data_dict[method] and metric in data_dict[method][cancer]:
                values[method].append(data_dict[method][cancer][metric])
            else:
                # Use NaN for missing data
                values[method].append(np.nan)
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(cancer_types)
    
    # Create angles for each cancer type (evenly distributed around the circle)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create a nice colormap
    colors = ['#8884d8', '#82ca9d']
    markers = ['o', 's']
    
    # Plot each method
    for i, method in enumerate(methods):
        # Get values for the method and extend to close the loop
        values_method = values[method] + [values[method][0]]
        
        # Plot the values
        ax.plot(angles, values_method, 'o-', linewidth=2, color=colors[i], 
                marker=markers[i], markersize=8, label=method)
        for j, (angle, value) in enumerate(zip(angles[:-1], values[method])):
                    ax.text(angle, value + 0.03, f"{value:.2f}", 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=10, color=colors[i])
                    
        ax.fill(angles, values_method, color=colors[i], alpha=0.1)
    
    # Set the y-axis limits
    ax.set_ylim(0, 1)
    
    # Fix axis to go in the right order and start at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels and title
    metric_names = {
        'auc_scores': 'AUC Score', 
        'f1': 'F1 Score', 
        'recall': 'Recall',
        'precision': 'Precision', 
        'accuracy': 'Accuracy'
    }
    
    plt.xticks(angles[:-1], cancer_types, size=12)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], 
               color="grey", size=10)
    
    # Add title
    plt.title(f'{feature_select} Comparison of {metric_names.get(metric, metric)} across Cancer Types', 
              size=20, pad=20)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Add gridlines
    ax.grid(True)
    
    # Add performance difference annotations
    for i, cancer in enumerate(cancer_types):
        ml_value = values['Synthetic'][i]
        deep_value = values['True'][i]
        
        # Calculate text position (midway between the points but a bit outside)
        angle = angles[i]
        better_method = 'Synthetic' if ml_value > deep_value else 'True'
        diff = abs(ml_value - deep_value)
        
        # Only annotate if there's a noticeable difference
        if diff > 0.05:
            # Calculate text position
            radius = max(ml_value, deep_value) + 0.05
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Add text annotation
            diff_text = f"+{diff:.2f}"
            color = colors[0] if better_method == 'Synthetic' else colors[1]
            ax.annotate(diff_text, xy=(angle, radius), xytext=(x, y),
                        textcoords='data', color=color, fontweight='bold',
                        horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax


def generate_all_metric_plots(data_dict, output_dir=None):
    """
    Generate radar plots for all metrics and save them.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing results for both methods.
    output_dir : str, optional
        Directory to save the figures. If None, figures are not saved.
    """
    metrics = ['auc_scores', 'f1', 'recall', 'precision', 'accuracy']
    
    for metric in metrics:
        fig, ax = radar_plot(data_dict, metric=metric)
        
        if output_dir:
            save_path = f"{output_dir}/radar_{metric}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()


# Example usage:
if __name__ == "__main__":
    # Sample data structure - replace with your actual results
    cancer_types = ["BRCA", "LUAD", "STAD", "HNSC", "LGG", "CESC", "LUSC", "ESCA", "GBM"]
    
    # Create a sample data dictionary
    data_dict = {
        'Synthetic': {
            'BRCA': {'auc_scores': 0.60, 'f1': 0.83, 'recall': 0.81, 'precision': 0.85, 'accuracy': 0.84},
            'LUAD': {'auc_scores': 0.58, 'f1': 0.78, 'recall': 0.79, 'precision': 0.77, 'accuracy': 0.79},
            'STAD': {'auc_scores': 0.73, 'f1': 0.79, 'recall': 0.77, 'precision': 0.81, 'accuracy': 0.80},
            'HNSC': {'auc_scores': 0.47, 'f1': 0.75, 'recall': 0.73, 'precision': 0.78, 'accuracy': 0.76},
            'LGG': {'auc_scores': 0.78, 'f1': 0.91, 'recall': 0.89, 'precision': 0.93, 'accuracy': 0.90},
            'CESC': {'auc_scores': 0.58, 'f1': 0.82, 'recall': 0.83, 'precision': 0.81, 'accuracy': 0.84},
            'LUSC': {'auc_scores': 0.56, 'f1': 0.74, 'recall': 0.72, 'precision': 0.76, 'accuracy': 0.75},
            'ESCA': {'auc_scores': 0.51, 'f1': 0.77, 'recall': 0.76, 'precision': 0.78, 'accuracy': 0.79},
            'GBM': {'auc_scores': 0.82, 'f1': 0.81, 'recall': 0.99, 'precision': 0.69, 'accuracy': 0.70}
        },
        'True': {
            'BRCA': {'auc_scores': 0.58, 'f1': 0.83, 'recall': 0.81, 'precision': 0.85, 'accuracy': 0.84},
            'LUAD': {'auc_scores': 0.50, 'f1': 0.78, 'recall': 0.79, 'precision': 0.77, 'accuracy': 0.79},
            'STAD': {'auc_scores': 0.69, 'f1': 0.79, 'recall': 0.77, 'precision': 0.81, 'accuracy': 0.80},
            'HNSC': {'auc_scores': 0.44, 'f1': 0.75, 'recall': 0.73, 'precision': 0.78, 'accuracy': 0.76},
            'LGG': {'auc_scores': 0.57, 'f1': 0.91, 'recall': 0.89, 'precision': 0.93, 'accuracy': 0.90},
            'CESC': {'auc_scores': 0.60, 'f1': 0.82, 'recall': 0.83, 'precision': 0.81, 'accuracy': 0.84},
            'LUSC': {'auc_scores': 0.57, 'f1': 0.74, 'recall': 0.72, 'precision': 0.76, 'accuracy': 0.75},
            'ESCA': {'auc_scores': 0.54, 'f1': 0.77, 'recall': 0.76, 'precision': 0.78, 'accuracy': 0.79},
            'GBM': {'auc_scores': 0.64, 'f1': 0.81, 'recall': 0.99, 'precision': 0.69, 'accuracy': 0.70}
        }
    }

    
    # Generate a radar plot for AUC scores
    radar_plot(data_dict, metric='auc_scores', save_path="/shares/sinha/sadeleye/ecPATH_Results/Synthetic_vs_True/Figures/ecPath_Direct_Gene_TitanGlobal.png")
    
    # Or generate plots for all metrics
    # generate_all_metric_plots(data_dict, output_dir='./plots')
