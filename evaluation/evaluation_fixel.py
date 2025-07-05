import numpy as np
import pandas as pd
from copy import deepcopy
from utils_metrics import load_fixel_data

def index_to_gt(gt, others, angle=45):
    n_gt = len(gt)
    indices = []
    errors = []
    for method in others:
        n_method = len(method)
        v1_u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
        v2_u = method / np.linalg.norm(method, axis=1, keepdims=True)
        angle_distance = np.arccos(np.clip(np.matmul(v1_u, v2_u.T), -1.0, 1.0))
        angle_index = -np.ones(n_gt, dtype=int)
        angular_error = -np.ones(n_gt, dtype=np.float32)
        indices.append(angle_index)
        errors.append(angular_error)
        if n_method > 0:
            mask = np.zeros((n_gt, n_method))
            masked_distance = np.ma.array(angle_distance, mask=mask)
            for i in range(n_gt):
                if masked_distance.count() and np.ma.min(masked_distance) < np.radians(angle):
                    flat_index = np.ma.argmin(masked_distance)
                    row, col = np.unravel_index(flat_index, angle_distance.shape)
                    angle_index[row] = col
                    angular_error[row] = angle_distance[row, col]
                    mask[row, :] = 1
                    mask[:, col] = 1
                    masked_distance = np.ma.array(masked_distance, mask=mask)
    return indices, errors

def fixel_comparison(target_indices, target_peak, target_afd, target_dir,
                     source_indices, source_peak, source_afd, source_dir, angle=45):
    angular_errors = []
    afd_sub_errors = []
    afdall_sub_errors = []
    peak_sub_errors = []
    peakall_sub_errors = []
    
    for m_index, m_peak, m_afd, m_dir in zip(source_indices, source_peak, source_afd, source_dir):
        indices, angular_error = index_to_gt(target_dir, m_dir, angle=angle)
        angular_errors.append(np.mean(angular_error))
        
        matched_afd_errors = [np.abs(m_afd[idx] - target_afd[i]) for i, idx in enumerate(indices) if idx >= 0]
        afd_sub_errors.append(np.mean(matched_afd_errors) if matched_afd_errors else 0)
        afdall_sub_errors.append(np.mean(np.concatenate([matched_afd_errors, m_afd])) if matched_afd_errors else np.mean(m_afd))
        
        matched_peak_errors = [np.abs(m_peak[idx] - target_peak[i]) for i, idx in enumerate(indices) if idx >= 0]
        peak_sub_errors.append(np.mean(matched_peak_errors) if matched_peak_errors else 0)
        peakall_sub_errors.append(np.mean(np.concatenate([matched_peak_errors, m_peak])) if matched_peak_errors else np.mean(m_peak))
    
    return {
        'afd err': np.mean(afd_sub_errors, axis=0),
        'afd extra err': np.mean(afdall_sub_errors, axis=0),
        'peak err': np.mean(peak_sub_errors, axis=0),
        'peak extra err': np.mean(peakall_sub_errors, axis=0),
        'angular err': np.mean(angular_errors, axis=0),
    }


def evaluate_fixel(gt_fixel_path, pred_fixel_paths, roi):
    gt_index, gt_afd, gt_peak, gt_dir = load_fixel_data(gt_fixel_path)
    pred_fixels = [load_fixel_data(p) for p in pred_fixel_paths]
    m_index, m_afd, m_peak, m_dir = zip(*pred_fixels)

    valid_gt = gt_index[roi]
    valid_m = [m[roi] for m in m_index]

    angular_errors, (afd_e, extra_afd_e, miss_afd_e), (peak_e, extra_peak_e, miss_peak_e) = fixel_comparison(
        valid_gt, gt_peak, gt_afd, gt_dir,
        valid_m, m_peak, m_afd, m_dir
    )

    def combined_mean(errors, extra, miss):
        return [
            np.mean([
                np.sum(e_i) + np.sum(e_e_i) + np.sum(m_i)
                for e_i, e_e_i, m_i in zip(e, e_e, m)
            ]) for e, e_e, m in zip(errors, extra, miss)
        ]

    metrics = {
        'angular_error': angular_errors,
        'afd_error': [np.mean(np.concatenate(e)) for e in afd_e],
        'afd_error_all': combined_mean(afd_e, extra_afd_e, miss_afd_e),
        'peak_error': [np.mean(np.concatenate(e)) for e in peak_e],
        'peak_error_all': combined_mean(peak_e, extra_peak_e, miss_peak_e),
    }
    return metrics


def aggregate_and_save_fixel_metrics(sub_list, method_names, metrics_list, output_path, filename_prefix):
    metrics_dict = {'sub': sub_list}
    for key in metrics_list[0].keys():
        metrics_dict[key] = [m[key] for m in metrics_list]
    df = pd.DataFrame(metrics_dict)
    df.to_csv(f"{output_path}/results_{filename_prefix}_fixel_metrics.csv", index=False)
    return df


def print_fixel_metrics(methods, metric_dict):
    def fmt(name, values, unit=''):
        string = ' | '.join([f'{v:04.2e}{unit}' for v in values])
        print(f'{name:<25} | {string}')

    print(f"\n{'Metric (Fixels)':<25} | " + ' | '.join([f"{m:>18}" for m in methods]))
    print('-' * (27 + len(methods) * 21))
    fmt("AFD error", metric_dict['afd_error'])
    fmt("AFD error (all)", metric_dict['afd_error_all'], unit='*')
    fmt("Peak error", metric_dict['peak_error'])
    fmt("Peak error (all)", metric_dict['peak_error_all'], unit='*')
    fmt("Angular error", metric_dict['angular_error'])


def summarize_fixel_metrics(metrics_list):
    return {
        key: np.mean([m[key] for m in metrics_list], axis=0)
        for key in metrics_list[0]
    }

