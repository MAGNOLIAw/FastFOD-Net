import os
import subprocess

def generate_fixel(result_path, method_name, test, suffix=None):
    """
    Generate fixel data using the specified method and test subjects.
    """
    fixel_path = os.path.join(result_path, method_name, 'fixel/')
    os.makedirs(fixel_path, exist_ok=True)
    
    for subject in test:
        sub_fod = os.path.join(result_path, method_name, f"{subject}{suffix or '.mif.gz'}")
        output_fixel = os.path.join(fixel_path, f"{subject}_fixel")
        subprocess.run(['fod2fixel', '-afd', 'afd.mif', '-peak_amp', 'peak.mif', sub_fod, output_fixel])

def main():
    test_subjects = [
        '211720', '672756', '161731', '130013', '144832', '133019', '103818', '208226', '160123', '199655',
        '298051', '159340', '239944', '149741', '113922', '103414', '129028', '105014', '127933', '140925'
    ]
    print(f'Total test subjects: {len(test_subjects)}')
    
    methods_list = ['unet_16dir_b2000', 'unet_20dir_b2000', 'unet_24dir_b2000']
    
    for method in methods_list:
        if method == '32dir_fod':
            result_path = '/home/data/HCP/'
            suffix = '_32dir_WMfod_norm.mif.gz'
        elif method == 'fod':
            result_path = '/home/data/HCP/'
            suffix = '_WMfod_norm.mif.gz'
        elif method.endswith(('b1000', 'b2000')) and not method.startswith('unet'):
            result_path = '/home/xinyi/HCP_data/'
            suffix = f'_WMfod_{method}_norm.mif.gz'
        else:
            result_path = '/home/xinyi/predictions_SR/hcp_training_hcp_testing/'
            suffix = None
        
        method_name = f"{method}/"
        generate_fixel(result_path, method_name, test_subjects, suffix)

if __name__ == "__main__":
    main()
