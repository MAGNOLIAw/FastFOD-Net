import os
import subprocess
import argparse
import sys

# ---------------------------- Config Section ---------------------------- #

# Method-specific configuration
# METHOD_CONFIG = {
#     '32dir_fod': {
#         'result_path': '/home/data/HCP/',
#         'suffix': '_32dir_WMfod_norm.mif.gz'
#     },
#     'fod': {
#         'result_path': '/home/data/HCP/',
#         'suffix': '_WMfod_norm.mif.gz'
#     },
#     # Add more methods as needed...
# }

# ---------------------------- Core Function ---------------------------- #

def generate_fixel(result_path, method_name, test_subjects, suffix=None):
    """
    Generate fixel data using the specified method and test subjects.
    """
    fixel_path = os.path.join(result_path, method_name, 'fixel')
    os.makedirs(fixel_path, exist_ok=True)

    for subject in test_subjects:
        sub_fod = os.path.join(result_path, method_name, f"{subject}{suffix or '.mif.gz'}")
        output_fixel = os.path.join(fixel_path, f"{subject}_fixel")
        print(f"[Processing] {subject} -> {output_fixel}")
        subprocess.run([
            'fod2fixel',
            '-afd', 'afd.mif',
            '-peak_amp', 'peak.mif',
            sub_fod,
            output_fixel
        ])

# ---------------------------- Entry Point ---------------------------- #

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate fixel data from FODs.")

    parser.add_argument('--method', type=str, required=True,
                        help='Method name (used as subfolder, e.g., unet_16dir_b2000)')

    parser.add_argument('--subjects', type=str, required=True,
                        help='Path to subject list file (one subject ID per line)')

    parser.add_argument('--result_path', type=str, required=True,
                        help='Base directory where FODs are stored')

    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix for FOD files, e.g., "_WMfod_norm.mif.gz" (default: ".mif.gz")')

    return parser.parse_args()

def main():
    args = parse_arguments()

    method = args.method
    subject_file = args.subjects
    result_path = args.result_path
    suffix = args.suffix

    if not os.path.exists(subject_file):
        print(f"[Error] Subject file not found: {subject_file}")
        sys.exit(1)

    with open(subject_file, 'r') as f:
        test_subjects = [line.strip() for line in f if line.strip()]

    print(f"[Info] Generating fixels for method: {method}")
    print(f"[Info] Total subjects: {len(test_subjects)}")
    print(f"[Info] Result path: {result_path}")
    print(f"[Info] Suffix: {suffix if suffix else '.mif.gz'}")

    generate_fixel(
        result_path=result_path,
        method_name=method,
        test_subjects=test_subjects,
        suffix=suffix
    )

if __name__ == "__main__":
    main()
