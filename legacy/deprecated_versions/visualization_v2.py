"""
V2 Visualization Script

This script generates supplementary visualizations for the
thesis report, such as the class imbalance and the
sequential aggregation bias.

Run this *after* all V2 model training is complete.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use the V2 config
import config_v2 as config 
import utils

logger = utils.logger

class ThesisVisualizer:
    
    def __init__(self):
        self.viz_dir = config.RESULT_PATHS['visualizations']
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Dynamically get the file names ---
        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        self.subset_name = "_".join(subset) if subset else "ALL"
        
        self.labeled_log_file = config.PROCESSED_DATA_DIR / f'processed_unified_logs_{self.subset_name}_LABELED.csv'
        self.static_features_file = config.PROCESSED_DATA_DIR / f'engineered_static_features_{self.subset_name}_v2.csv'
        self.sequence_labels_file = config.PROCESSED_DATA_DIR / f'sequence_labels_{self.subset_name}_v2.npy'
        
        # Set plot style
        plt.style.use(config.VISUALIZATION['style'])
        plt.rcParams['figure.figsize'] = config.VISUALIZATION['figure_size']
        plt.rcParams['figure.dpi'] = config.VISUALIZATION['dpi']
        self.colors = sns.color_palette(config.VISUALIZATION['color_palette'])

    def plot_class_imbalance(self):
        """
        Generates a bar chart showing the extreme class imbalance in the
        raw event data.
        """
        logger.info(f"Generating class imbalance plot from {self.labeled_log_file.name}...")
        try:
            # We don't need to load the whole file, we can just iterate
            # and count the labels.
            total_rows = 0
            insider_rows = 0
            
            with pd.read_csv(self.labeled_log_file, chunksize=2000000, usecols=['is_insider']) as reader:
                for chunk in reader:
                    total_rows += len(chunk)
                    insider_rows += chunk['is_insider'].sum()
            
            benign_rows = total_rows - insider_rows
            
            data = {'Class': ['Benign Events', 'Malicious Events'],
                    'Count': [benign_rows, insider_rows]}
            df = pd.DataFrame(data)
            
            plt.figure()
            ax = sns.barplot(x='Class', y='Count', data=df, palette=[self.colors[0], self.colors[1]])
            ax.set_yscale('log') # Use a log scale so the malicious bar is visible
            ax.set_title(f'Class Imbalance (Log Scale) - V2 Subset ({self.subset_name})\nTotal Events: {total_rows:,}')
            ax.set_ylabel('Event Count (Log Scale)')
            ax.set_xlabel('Event Class')
            
            # Add labels
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height()):,}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 9), 
                            textcoords='offset points')
            
            save_path = self.viz_dir / f"class_imbalance_{self.subset_name}_v2.png"
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Class imbalance plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Could not generate class imbalance plot: {e}")

    def plot_sequential_aggregation_bias(self):
        """
        Generates a bar chart visualizing the number of insiders
        lost during the sequence generation process.
        """
        logger.info("Generating Sequential Aggregation Bias plot...")
        try:
            # 1. Get insiders from the *static* file (this is the "truth")
            static_df = pd.read_csv(self.static_features_file)
            static_insiders = static_df['is_insider'].sum()
            
            # 2. Get insiders from the *sequence* file
            sequence_labels = np.load(self.sequence_labels_file)
            sequence_insiders = sequence_labels.sum()
            
            data = {'Model Type': ['Static (IF, DC)', 'Sequential (LSTM)'],
                    'Insider Samples': [static_insiders, sequence_insiders]}
            df = pd.DataFrame(data)

            plt.figure()
            ax = sns.barplot(x='Model Type', y='Insider Samples', data=df, palette=[self.colors[2], self.colors[3]])
            ax.set_title(f'Insider Sample Loss During Sequence Generation\n(Sequence Length = {config.SEQUENCE_LENGTH} days)')
            ax.set_ylabel('Number of Insider Samples in Test Set')
            ax.set_xlabel('Model Data Source')
            
            # Add labels
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 9), 
                            textcoords='offset points')
            
            save_path = self.viz_dir / f"sequential_bias_{self.subset_name}_v2.png"
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Sequential bias plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Could not generate sequential bias plot: {e}")

    def run_all(self):
        logger.info(utils.generate_report_header("V2 THESIS VISUALIZATION SCRIPT"))
        self.plot_class_imbalance()
        self.plot_sequential_aggregation_bias()
        logger.info("All supplementary visualizations created.")

def main():
    visualizer = ThesisVisualizer()
    visualizer.run_all()

if __name__ == "__main__":
    main()