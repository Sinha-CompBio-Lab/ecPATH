import os
import json
from data_plots import radar_plot 

def load_all_results(input_dir="results"):
    """
    Load results from all text files and organize them by data type and cancer type.
    """
    # Initialize the combined dictionary
    combined_dict = {
        'Synthetic': {},
        'True': {}
    }
    
    # Check if the directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist")
        return combined_dict
    
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('_results.txt')]
    
    # Read each file and update the combined dictionary
    for file_name in txt_files:
        file_path = os.path.join(input_dir, file_name)
        
        # Extract cancer type from file name
        cancer_type = file_name.split('_results.txt')[0]
        
        try:
            # Load the data from the file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Update the combined dictionary
            if 'Synthetic' in data and cancer_type:
                combined_dict['Synthetic'][cancer_type] = data['Synthetic'][cancer_type]
            
            if 'True' in data and cancer_type:
                combined_dict['True'][cancer_type] = data['True'][cancer_type]
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return combined_dict


if __name__ == "__main__":
    feature_select = "uni_Test"
    data_dict = load_all_results(f"/shares/sinha/sadeleye/ecPATH_Results/Synthetic_vs_True/{feature_select}")

    # Generate a radar plot for AUC scores
    radar_plot(data_dict, metric='auc_scores', save_path=f"/shares/sinha/sadeleye/ecPATH_Results/Synthetic_vs_True/Figures/ecPath_Synthetic_True_Auc_{feature_select}.png",feature_select=feature_select)
    # radar_plot(data_dict, metric='f1', save_path=f"/shares/sinha/sadeleye/ecPATH_Results/Synthetic_vs_True/Figures/ecPath_Synthetic_True_F1_{feature_select}.png",feature_select=feature_select)
    