import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils_metrics import load_connectome, generate_data_dict
from connectome import (
    significant_edges_t, significant_edges_w, graph_tests,
    disparity_matrix, ranking_metrics, graph_str_eff_cpath,
    graph_str_eff_cpath_ratio, triu_to_graph
)
from utils_metrics import plot_correlation

def main():
    # Configuration
    main_path = '/home/xinyi/MSBIR/'
    result_path = '/home/xinyi/predictions_SR/msbir_training_msbir_testing/'
    methods = ['30dir_b1000']
    experiment = '_'.join(methods)

    # Subject List
    fod_dir = os.path.join(main_path, 'fod_bbox')
    sub_list = [f[:-len("_WMfod_norm.mif.gz")] for f in os.listdir(fod_dir) if f.endswith("_WMfod_norm.mif.gz")]
    excludes = {'AU-025-1674_201010-bl', 'AU-025-0037_220326-bl', 'AU-025-0231_200312-bl'}
    sub_list = sorted(s for s in sub_list if s not in excludes)

    # Data Dictionary
    data_dict = generate_data_dict(
        main_path, 'fod_vox1.25', 'connectome', methods,
        'brainmask', 'wm', result_path,
        subjects=sub_list, layers=None, load_connectome='connectome'
    )

    # Load Connectomes
    conn_list, triu_list = [], []
    for data in data_dict.values():
        gt_conn, gt_triu = load_connectome(data['gt']['connectome'])
        method_conns = [gt_conn] + [load_connectome(data['methods'][m]['connectome'])[0] for m in methods]
        method_trius = [gt_triu] + [load_connectome(data['methods'][m]['connectome'])[1] for m in methods]
        conn_list.append(np.array(method_conns))
        triu_list.append(np.array(method_trius))

    all_conn = np.stack(conn_list, axis=1)
    all_triu = np.stack(triu_list, axis=1)

    # Compute Metrics
    pval_t, fdr_t, fwe_t, _ = significant_edges_t(all_triu[0], all_triu[1:])
    pval_w, fdr_w, fwe_w, _ = significant_edges_w(all_triu[0], all_triu[1:])

    str_t, str_w, eff_t, eff_w, _, _ = graph_tests(
        all_conn[0], all_conn[1:], save_tocsv=result_path,
        sub_list=sub_list, method_list=methods
    )

    disp = disparity_matrix(all_conn[0], all_conn[1:])
    diff_str, diff_eff, _ = graph_str_eff_cpath(all_conn[0], all_conn[1:])
    diff_str_ratio, diff_eff_ratio, _ = graph_str_eff_cpath_ratio(all_conn[0], all_conn[1:])

    inter, intra = ranking_metrics(all_triu[0], all_triu[1:])
    inter_taus, inter_wtaus, inter_iwtaus, inter_pvals = inter
    intra_taus, intra_wtaus, intra_iwtaus, intra_pvals = intra

    # Save Metrics
    r, c = np.triu_indices(all_conn.shape[-1], 1)
    df = pd.DataFrame({
        'methods': methods,
        'dconn': [np.sum(d[r, c]) for d in disp],
        'mdconn': [np.mean(d[r, c]) for d in disp],
        'tau_mean': [np.mean(t) for t in inter_taus],
        'tau_std': [np.std(t) for t in inter_taus],
        'tau_pval': inter_pvals,
        'wtau_mean': [np.mean(t) for t in inter_wtaus],
        'wtau_std': [np.std(t) for t in inter_wtaus],
        'iwtau_mean': [np.mean(t) for t in inter_iwtaus],
        'iwtau_std': [np.std(t) for t in inter_iwtaus]
    })
    df.to_csv(os.path.join(result_path, f'{experiment}_metrics_connectome.csv'), index=False)

    # Plot Correlation
    plt.figure(figsize=(len(methods) * 6, 5))
    for i, method in enumerate(methods):
        plt.subplot(1, len(methods), i + 1)
        plot_correlation(all_triu[i + 1].flatten(), all_triu[0].flatten(), method, 'gt')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'{experiment}_correlation.png'))

    # Plot P-value Heatmaps
    plt.figure(figsize=(len(methods) * 6, 10))
    for i, method in enumerate(methods):
        plt.subplot(2, len(methods), i + 1)
        sns.heatmap(triu_to_graph(1 - pval_t[i]), cmap='jet', vmin=0.95, vmax=1, square=True, xticklabels=False, yticklabels=False)
        plt.title(f'T-test (1-p) [{method}]')
        plt.subplot(2, len(methods), len(methods) + i + 1)
        sns.heatmap(triu_to_graph(1 - pval_w[i]), cmap='jet', vmin=0.95, vmax=1, square=True, xticklabels=False, yticklabels=False)
        plt.title(f'Wilcoxon (1-p) [{method}]')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'{experiment}_pvalue_maps.png'))

if __name__ == '__main__':
    main()
