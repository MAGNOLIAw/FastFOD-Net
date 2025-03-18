import numpy as np

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
