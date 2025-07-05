import numpy as np
import pandas as pd

def mean_squared_error(target, source, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]
    return np.sum((target - source) ** 2, axis=1)

def mean_absolute_error(target, source, roi=None):
    if roi is not None:
        source = source[roi.astype(bool)]
        target = target[roi.astype(bool)]
    return np.sum(np.abs(target - source), axis=1)

def psnr(target, source, roi=None, pixel_max=1):
    mse = mean_squared_error(source, target, roi)
    mse_norm = mse / mse.size
    mse_mask = mse < 1.0e-10
    psnr_image = 20 * np.log10(pixel_max / np.sqrt(mse_norm))
    psnr_image[mse_mask] = 100
    return psnr_image

def angular_correlation(target, source, roi=None):
    if roi is None:
        pred_fod = source[..., 1:]
        gt_fod = target[..., 1:]
    else:
        pred_fod = source[roi.astype(bool), 1:]
        gt_fod = target[roi.astype(bool), 1:]
    
    numerator = np.sum(pred_fod * gt_fod, axis=-1)
    denominator = np.sqrt(np.sum(pred_fod ** 2, axis=-1)) * np.sqrt(np.sum(gt_fod ** 2, axis=-1))
    return numerator / denominator

def fod_comparison(target_fod, source_fods, roi=None):
    mse_list = []
    mae_list = []
    psnr_list = []
    acc_list = []
    for m_fod_i in source_fods:
        mse_list.append(mean_squared_error(target_fod, m_fod_i, roi))
        mae_list.append(mean_absolute_error(target_fod, m_fod_i, roi))
        psnr_list.append(psnr(target_fod, m_fod_i, roi))
        acc_list.append(angular_correlation(target_fod, m_fod_i, roi))
    
    return mse_list, mae_list, psnr_list, acc_list

def evaluate_fod(gt_fod, pred_fods, roi):
    mse_list, mae_list, psnr_list, acc_list = fod_comparison(gt_fod, pred_fods, roi)
    metrics = {
        'mse': [np.mean(e) for e in mse_list],
        'mae': [np.mean(e) for e in mae_list],
        'psnr': [np.mean(e) for e in psnr_list],
        'acc': [np.nanmean(e) for e in acc_list],
    }
    return metrics


def aggregate_and_save_fod_metrics(sub_list, method_names, metrics_list, output_path, filename_prefix):
    metrics_dict = {'sub': sub_list}
    for key in metrics_list[0].keys():
        metrics_dict[f'fod_{key}'] = [m[key] for m in metrics_list]
    df = pd.DataFrame(metrics_dict)
    df.to_csv(f"{output_path}/results_{filename_prefix}_fod_metrics.csv", index=False)
    return df


def print_fod_metrics(methods, metric_dict):
    def fmt(name, values, unit=''):
        string = ' | '.join([f'{v:04.2e}{unit}' if unit != 'dB' else f'{v:05.2f} {unit}' for v in values])
        print(f'{name:<25} | {string}')

    print(f"\n{'Metric (FODs)':<25} | " + ' | '.join([f"{m:>18}" for m in methods]))
    print('-' * (27 + len(methods) * 21))
    fmt("Mean absolute error", metric_dict['mae'])
    fmt("Mean squared error", metric_dict['mse'])
    fmt("PSNR (dB)", metric_dict['psnr'], unit='dB')
    fmt("Angular correlation", metric_dict['acc'])


def summarize_fod_metrics(metrics_list):
    return {
        key: np.mean([m[key] for m in metrics_list], axis=0)
        for key in metrics_list[0]
    }


