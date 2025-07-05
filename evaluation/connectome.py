import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import kendalltau, weightedtau

from graph_metrics import strengths_und, efficiency_wei, charpath

import os
import pandas as pd

from statsmodels.stats.multitest import multipletests

def pvalue_percentages(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    fdr = fdr_corrected_percentage(pvalues, alpha)
    fwe = fwe_corrected_percentage(pvalues, alpha)
    nocor = 100 * np.mean(pvalues < alpha)

    return fdr, fwe, nocor


def pvalue_percentages_v2(pvalues, alpha):
    """
    Function to calculate FDR, FWE, and no correction percentages.
    :param pvalues:
    :param alpha:
    :return:
    """
    # This is a placeholder function. Implement your actual logic here.
    fdr_corrected, _, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')
    fdr = 100 * np.mean(fdr_corrected)
    fwe_corrected, _, _, _ = multipletests(pvalues, alpha=alpha, method='bonferroni')
    fwe = 100 * np.mean(fwe_corrected)  # This is just an example
    nocor = 100 * np.mean(pvalues < alpha)  # This is just an example
    return fdr, fwe, nocor


def fdr_corrected_percentage(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    m = len(pvalues)
    k_all = np.array(list(range(1, m + 1)))
    reject_list = np.where(
        np.sort(pvalues) <= (k_all * alpha / m)
    )[0]
    if len(reject_list) > 0:
        k = reject_list[-1]
    else:
        k = 0

    return 100 * k / m

def fdr_corrected_percentage_connections(pvalues, alpha=0.01):
    """
    Function to calculate the percentage of p-values that are significant after FDR correction.
    :param pvalues: array-like, p-values from hypothesis tests.
    :param alpha: float, significance level.
    :return: float, percentage of significant p-values, and array of significant connections.
    """
    m = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.sort(pvalues)
    k_all = np.array(list(range(1, m + 1)))

    # Find the largest k for which the p-value is less than (k * alpha / m)
    reject_list = np.where(
        sorted_pvalues <= (k_all * alpha / m)
    )[0]

    if len(reject_list) > 0:
        k = reject_list[-1]
        threshold = sorted_pvalues[k]
    else:
        k = 0
        threshold = 0

    # Create a boolean array for significant connections
    significant_connections = pvalues <= threshold

    percentage = 100 * np.sum(significant_connections) / m

    return percentage, significant_connections


def fwe_corrected_percentage(pvalues, alpha=0.01):
    """

    :param pvalues:
    :param alpha:
    :return:
    """
    m = len(pvalues)
    bonf_alpha = alpha / m

    return 100 * np.mean(pvalues < bonf_alpha)


def significant_edges_t(target, source, alpha=0.01):
    """
    Function to check significantly different edges (no check for which is
     higher).
    :param target:
    :param source:
    :param alpha:
    :return:
    """

    fdr_list = []
    fwe_list = []
    nocor_list = []
    pvalues_list = []
    for m in source: # for each method
        _, pvalues = ttest_rel(target, m, axis=0)
        novar_mask = np.isnan(pvalues)
        pvalues[novar_mask] = 1

        fdr, fwe, nocor = pvalue_percentages(pvalues, alpha)

        print('fdr, fwe, nocor', fdr, fwe, nocor)

        fdr_list.append(fdr)
        fwe_list.append(fwe)
        nocor_list.append(nocor)
        pvalues_list.append(pvalues)

        # print('pvalues', pvalues, len(pvalues)) # 3486 = number of edges

    return pvalues_list, fdr_list, fwe_list, nocor_list


def compute_effect_size(target, source):
    """
    Function to compute Cohen's d for each edge between target and source.
    """
    mean_diff = np.mean(target - source, axis=0)
    pooled_std = np.sqrt((np.std(target, axis=0, ddof=1) ** 2 + np.std(source, axis=0, ddof=1) ** 2) / 2)

    # Handle the case where pooled_std is zero to avoid division by zero
    pooled_std[pooled_std == 0] = np.nan
    effect_size = mean_diff / pooled_std

    return effect_size

def significant_edges_t_corrected(target, source, alpha=0.01):
    """
    Function to check significantly different edges (no check for which is
    higher).
    :param target:
    :param source:
    :param alpha:
    :return:
    """

    fdr_list = []
    fwe_list = []
    nocor_list = []
    pvalues_list = []
    corrected_pvalues_list = []
    rejected_list = []
    effect_sizes_list = []

    for m in source: # for each method
        # print(np.array(target).shape, np.array(m).shape) # (20, 3486) (20, 3486)
        _, pvalues = ttest_rel(target, m, axis=0)
        novar_mask = np.isnan(pvalues)
        pvalues[novar_mask] = 1

        # Apply FDR correction using Benjamini-Hochberg procedure
        rejected, corrected_pvalues, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')
        # corrected_pvalues[novar_mask] = 1  # Handle NaNs in corrected p-values

        # Calculate effect sizes for each edge
        effect_sizes = compute_effect_size(target, m)

        fdr, fwe, nocor = pvalue_percentages_v2(pvalues, alpha)

        fdr_list.append(fdr)
        fwe_list.append(fwe)
        nocor_list.append(nocor)
        pvalues_list.append(pvalues)
        corrected_pvalues_list.append(corrected_pvalues)
        rejected_list.append(rejected)  # Save the significance after correction
        effect_sizes_list.append(effect_sizes)

        print('fdr, fwe, nocor v2', fdr, fwe, nocor)

        # print('pvalues', pvalues, len(pvalues)) # 3486 = number of edges

    return pvalues_list, corrected_pvalues_list, rejected_list, fdr_list, fwe_list, nocor_list, effect_sizes_list


def significant_edges_w(target, source, alpha=0.01):
    """
    Function to check significantly different edges (no check for which is
     higher).
    :param target:
    :param source:
    :param alpha:
    :return:
    """

    fdr_list = []
    fwe_list = []
    nocor_list = []
    pvalues_list = []
    for m in source:
        pvalues = np.array([
            wilcoxon(gs_i, pipe_i, 'zsplit')[1]
            for gs_i, pipe_i in zip(target.transpose(), m.transpose())
        ])
        novar_mask = np.isnan(pvalues)
        pvalues[novar_mask] = 1

        fdr, fwe, nocor = pvalue_percentages(pvalues, alpha)

        fdr_list.append(fdr)
        fwe_list.append(fwe)
        nocor_list.append(nocor)
        pvalues_list.append(pvalues)

    return pvalues_list, fdr_list, fwe_list, nocor_list


def conn_mask(edge_list, edges):
    """

    :param edge_list:
    :param edges:
    :return:
    """
    edge_mask = np.zeros(edges)
    edge_mask[edge_list] = 1

    return triu_to_graph(edge_mask)


def triu_to_graph(triu):
    """

    :param triu:
    :return:
    """
    n_edges = len(triu)
    nodes = int((np.sqrt(8 * n_edges + 1) + 1) / 2)
    r, c = np.triu_indices(nodes, 1)

    graph = np.zeros((nodes, nodes))
    graph[r, c] = triu
    graph[c, r] = triu

    return graph



def graph_tests(target, source, save_tocsv=None, sub_list=None, method_list=None):
    relstr_t = []
    releff_t = []
    relcpath_t = []
    relstr_w = []
    releff_w = []
    relcpath_w = []

    gt_strength = np.mean([
        strengths_und(conn) for conn in target
    ], axis=-1)
    gt_efficiency = np.array([
        efficiency_wei(conn) for conn in target
    ])
    gt_charpath = np.array([
        charpath(conn)[0] for conn in target
    ])

    # print('gt_strength', gt_strength.shape)  # (111,) (n_samples,)
    if save_tocsv != None:
        res = {
            'sub_list': sub_list,
            'strength': gt_strength,
            'efficiency': gt_efficiency
        }
        output_path = save_tocsv
        # name = 'gt'
        experiment_name = f'gt_metrics_graph'
        results_csv_file = f'results_{experiment_name}.csv'
        if not os.path.exists(output_path):
            subprocess.call(['mkdir', output_path])
        metrics_pd = pd.DataFrame(res)
        metrics_pd.to_csv(os.path.join(output_path, results_csv_file))

    for m_i, m in enumerate(source):
        m_strength = np.mean([
            strengths_und(conn) for conn in m
        ], axis=-1)
        m_efficiency = np.array([
            efficiency_wei(conn) for conn in m
        ])
        m_charpath = np.array([
            charpath(conn)[0] for conn in m
        ])

        _, pvalue = ttest_rel(gt_strength, m_strength)
        relstr_t.append(pvalue)
        _, pvalue = wilcoxon(gt_strength, m_strength)
        relstr_w.append(pvalue)

        _, pvalue = ttest_rel(gt_efficiency, m_efficiency)
        releff_t.append(pvalue)
        _, pvalue = wilcoxon(gt_efficiency, m_efficiency)
        releff_w.append(pvalue)

        # _, pvalue = ttest_rel(gt_charpath, m_charpath)
        # relcpath_t.append(pvalue)
        # _, pvalue = wilcoxon(gt_charpath, m_charpath)
        # relcpath_w.append(pvalue)

        if save_tocsv != None:
            res = {
                'sub_list': sub_list,
                'strength': m_strength,
                'efficiency': m_efficiency
            }
            output_path = save_tocsv
            name = method_list[m_i]
            experiment_name = f'{method_list[m_i]}_metrics_graph'
            results_csv_file = f'results_{experiment_name}.csv'
            if not os.path.exists(output_path):
                subprocess.call(['mkdir', output_path])
            metrics_pd = pd.DataFrame(res)
            metrics_pd.to_csv(os.path.join(output_path, results_csv_file))

    return relstr_t, relstr_w, releff_t, releff_w, relcpath_t, relcpath_w

def graph_str_eff_cpath(target, source):
    relstr_t = []
    releff_t = []
    relcpath_t = []
    relstr_w = []
    releff_w = []
    relcpath_w = []

    diffstr = []
    diffeff = []
    diffcpath = []

    gt_strength = np.mean([
        strengths_und(conn) for conn in target
    ], axis=-1)
    gt_efficiency = np.array([
        efficiency_wei(conn) for conn in target
    ])
    gt_charpath = np.array([
        charpath(conn)[0] for conn in target
    ])
    for m in source:
        m_strength = np.mean([
            strengths_und(conn) for conn in m
        ], axis=-1)
        m_efficiency = np.array([
            efficiency_wei(conn) for conn in m
        ])
        m_charpath = np.array([
            charpath(conn)[0] for conn in m
        ])

        # print('m', m_strength, m_efficiency, m_charpath)

        diff = gt_strength - m_strength
        diffstr.append(diff)

        diff = gt_efficiency - m_efficiency
        diffeff.append(diff)

        # diff = gt_charpath - m_charpath
        # diffcpath.append(diff)

        _, pvalue = ttest_rel(gt_strength, m_strength)
        relstr_t.append(pvalue)
        _, pvalue = wilcoxon(gt_strength, m_strength)
        relstr_w.append(pvalue)

        _, pvalue = ttest_rel(gt_efficiency, m_efficiency)
        releff_t.append(pvalue)
        _, pvalue = wilcoxon(gt_efficiency, m_efficiency)
        releff_w.append(pvalue)

        # _, pvalue = ttest_rel(gt_charpath, m_charpath)
        # relcpath_t.append(pvalue)
        # _, pvalue = wilcoxon(gt_charpath, m_charpath)
        # relcpath_w.append(pvalue)

    return diffstr, diffeff, diffcpath

def graph_str_eff_cpath_ratio(target, source):
    relstr_t = []
    releff_t = []
    relcpath_t = []
    relstr_w = []
    releff_w = []
    relcpath_w = []

    diffstr = []
    diffeff = []
    diffcpath = []

    gt_strength = np.mean([
        strengths_und(conn) for conn in target
    ], axis=-1)
    gt_efficiency = np.array([
        efficiency_wei(conn) for conn in target
    ])
    gt_charpath = np.array([
        charpath(conn)[0] for conn in target
    ])
    for m in source:
        m_strength = np.mean([
            strengths_und(conn) for conn in m
        ], axis=-1)
        m_efficiency = np.array([
            efficiency_wei(conn) for conn in m
        ])
        m_charpath = np.array([
            charpath(conn)[0] for conn in m
        ])

        # print('m', m_strength, m_efficiency, m_charpath)

        diff = (gt_strength - m_strength)/gt_strength
        diffstr.append(100 * diff)

        diff = (gt_efficiency - m_efficiency)/gt_efficiency
        diffeff.append(100 * diff)

        # diff = (gt_charpath - m_charpath)/gt_charpath
        # diffcpath.append(100 * diff)

        # _, pvalue = ttest_rel(gt_strength, m_strength)
        # relstr_t.append(pvalue)
        # _, pvalue = wilcoxon(gt_strength, m_strength)
        # relstr_w.append(pvalue)
        #
        # _, pvalue = ttest_rel(gt_efficiency, m_efficiency)
        # releff_t.append(pvalue)
        # _, pvalue = wilcoxon(gt_efficiency, m_efficiency)
        # releff_w.append(pvalue)
        #
        # _, pvalue = ttest_rel(gt_charpath, m_charpath)
        # relcpath_t.append(pvalue)
        # _, pvalue = wilcoxon(gt_charpath, m_charpath)
        # relcpath_w.append(pvalue)

    return diffstr, diffeff, diffcpath

def disparity_matrix(target, source):
    disp = []
    for m in source: # for each method
        disp.append(np.mean(np.abs(target - m), axis=0))

    return disp

def mean_matrix(all):
    meanm = []
    for m in all: # for each method
        meanm.append(np.mean(m, axis=0))
    return meanm


def ranking_metrics(target, source, alpha=0.01):
    """

    :param target:
    :param source:
    :return:
    """

    m = target.shape[-1]
    bonf_alpha = alpha / m

    # Comparison between the gold standard and other methods.
    inter_taus = []
    inter_wtaus = []
    inter_iwtaus = []
    inter_pvalues = []
    for m in source:
        inter_taus.append(
            [kendalltau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)] # for each subject
        )
        inter_wtaus.append(
            [weightedtau(gs_sub, pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)]
        )
        inter_iwtaus.append(
            [weightedtau(-gs_sub, -pipe_sub)[0] for gs_sub, pipe_sub in zip(target, m)]
        )
        inter_pvalues.append(
            np.mean([
                kendalltau(gs_sub, pipe_sub)[1] < bonf_alpha
                for gs_sub, pipe_sub in zip(target, m)
            ])
        )

    inter_corr = [inter_taus, inter_wtaus, inter_iwtaus, inter_pvalues]

    # Comparison between gold standard inviduals.
    # This is a bit different from the previous set of metrics. Here we want
    # to see how correlated are the rankings between subjects. Ideally, this
    # should be lower than the correlation between methods. Otherwise, the
    # "errors" would be higher than the real variability between individuals.
    intra_taus = []
    intra_wtaus = []
    intra_iwtaus = []
    intra_pvalues = []
    for i in range(0, len(target)):
        for j in range(i + 1, len(target)):
            i_conn = target[i, :]
            j_conn = target[j, :]
            tau, pvalue = kendalltau(i_conn, j_conn)
            intra_taus.append(tau)
            intra_pvalues.append(pvalue < bonf_alpha)
            tau, _ = weightedtau(i_conn, j_conn)
            intra_wtaus.append(tau)
            tau, _ = weightedtau(-i_conn, -j_conn)
            intra_iwtaus.append(tau)

    intra_corr = [intra_taus, intra_wtaus, intra_iwtaus, intra_pvalues]

    return inter_corr, intra_corr
