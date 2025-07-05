# main_fixel.py
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from evaluation_fixel import evaluate_fixel, aggregate_and_save_fixel_metrics, print_fixel_metrics, summarize_fixel_metrics
from utils_metrics import generate_data_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fixel metrics.")
    parser.add_argument('--dataset', type=str, choices=['hcp', 'mnd', 'ms', 'mrgfus'], required=True)
    parser.add_argument('--roi_type', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+', required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--angle', type=int, default=45)
    parser.add_argument('--pickle_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    return parser.parse_args()

def load_subjects(pickle_path):
    with open(pickle_path, 'rb') as f:
        sub_data = pickle.load(f)
    return list(sub_data.keys()) if isinstance(sub_data, dict) else sub_data

def evaluate_fixel_all(data_dict, roi_type, angle, name, output_path):
    fixel_metrics_all = []
    subject_list = list(data_dict.keys())

    for i, (sub, sub_data) in enumerate(data_dict.items()):
        print(f"[{i+1}/{len(subject_list)}] Subject: {sub}")
        roi = sub_data['roi'].astype(bool)
        gt_fixel = sub_data['gt']['fixel']
        pred_fixels = [m['fixel'] for m in sub_data['methods'].values()]
        fixel_metrics = evaluate_fixel(gt_fixel, pred_fixels, roi)
        fixel_metrics_all.append(fixel_metrics)

    methods = list(sub_data['methods'].keys())
    angle_prefix = f"{angle}_{roi_type}_{name}".replace("_", "")
    os.makedirs(output_path, exist_ok=True)

    fixel_df = aggregate_and_save_fixel_metrics(subject_list, methods, fixel_metrics_all, output_path, angle_prefix)
    fixel_means = summarize_fixel_metrics(fixel_metrics_all)
    print_fixel_metrics(methods, fixel_means)

    summary_df = pd.DataFrame({'methods': methods, **fixel_means})
    summary_df.to_csv(os.path.join(output_path, f"results_{angle_prefix}_fixel_summary.csv"), index=False)

if __name__ == '__main__':
    args = parse_args()

    dataset_paths = {
        'hcp': '/home/xinyi/HCPanomalies_Database/',
        'mnd': '/home/xinyi/MND/',
        'ms': '/home/xinyi/MSBIR/',
        'mrgfus': '/home/xinyi/MRgFUS/'
    }
    main_path = dataset_paths[args.dataset]
    wm_path = args.roi_type if not args.roi_type.startswith('ROI') else 'purewm'

    sub_list = load_subjects(args.pickle_path)
    data_dict = generate_data_dict(
        main_path, gt_conn_path='connectome',
        method_list=args.methods,
        brain_path='brainmask',
        wm_path=wm_path,
        result_path=args.result_path,
        subjects=sub_list
    )

    evaluate_fixel_all(data_dict, args.roi_type, args.angle, args.name, os.path.join(args.result_path, 'metrics_fixel'))
