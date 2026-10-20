"""Compute Maximum Mean Discrepancy (MMD) between semantic distributions.

Uses RBF kernel with median heuristic for bandwidth selection.
Implements unbiased MMD estimator with equal-size sampling.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))

    Args:
        X: (m, d) array
        Y: (n, d) array
        sigma: bandwidth parameter

    Returns:
        K: (m, n) kernel matrix
    """
    # Compute pairwise squared distances
    dists = cdist(X, Y, metric='sqeuclidean')
    K = np.exp(-dists / (2 * sigma ** 2))
    return K


def median_heuristic(X: np.ndarray, Y: np.ndarray = None, subsample: int = 2000) -> float:
    """Compute median heuristic for bandwidth selection.

    sigma = median(pairwise distances)

    Args:
        X: (m, d) array
        Y: (n, d) array or None (if None, use X)
        subsample: max number of samples to use for efficiency

    Returns:
        sigma: bandwidth parameter
    """
    if Y is None:
        # Within-dataset distances
        if len(X) > subsample:
            idx = np.random.choice(len(X), subsample, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        dists = pdist(X_sub, metric='euclidean')
    else:
        # Between-dataset distances
        if len(X) > subsample:
            idx_x = np.random.choice(len(X), subsample, replace=False)
            X = X[idx_x]
        if len(Y) > subsample:
            idx_y = np.random.choice(len(Y), subsample, replace=False)
            Y = Y[idx_y]
        dists = cdist(X, Y, metric='euclidean').flatten()

    sigma = float(np.median(dists))
    return sigma


def mmd_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Compute unbiased MMD^2 estimator.

    MMD²(X,Y) = 1/[m(m-1)] * sum_{i!=j} k(x_i, x_j)
                + 1/[n(n-1)] * sum_{i!=j} k(y_i, y_j)
                - 2/(mn) * sum_{i,j} k(x_i, y_j)

    Args:
        X: (m, d) array
        Y: (n, d) array
        sigma: bandwidth parameter

    Returns:
        mmd2: unbiased MMD^2 estimate (can be slightly negative due to finite sample)
    """
    m = len(X)
    n = len(Y)

    # Within X
    K_xx = rbf_kernel(X, X, sigma)
    # Remove diagonal
    np.fill_diagonal(K_xx, 0)
    term1 = K_xx.sum() / (m * (m - 1))

    # Within Y
    K_yy = rbf_kernel(Y, Y, sigma)
    np.fill_diagonal(K_yy, 0)
    term2 = K_yy.sum() / (n * (n - 1))

    # Between X and Y
    K_xy = rbf_kernel(X, Y, sigma)
    term3 = K_xy.sum() / (m * n)

    mmd2 = term1 + term2 - 2 * term3

    return mmd2


def compute_mmd_with_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    sample_size: int,
    num_repeats: int = 100,
    seed: int = 42
) -> dict[str, float]:
    """Compute MMD with repeated equal-size sampling.

    Args:
        X: (m, d) embeddings from source A
        Y: (n, d) embeddings from source B
        sample_size: number of samples to draw from each source per repeat
        num_repeats: number of bootstrap repeats
        seed: random seed

    Returns:
        dict with mean, std, min, max, quantiles of MMD
    """
    np.random.seed(seed)

    mmd_values = []
    sigmas = []

    for _ in range(num_repeats):
        # Sample equal size from both
        idx_x = np.random.choice(len(X), sample_size, replace=False)
        idx_y = np.random.choice(len(Y), sample_size, replace=False)

        X_sample = X[idx_x]
        Y_sample = Y[idx_y]

        # Compute bandwidth
        sigma = median_heuristic(X_sample, Y_sample)
        sigmas.append(sigma)

        # Compute MMD
        mmd2 = mmd_unbiased(X_sample, Y_sample, sigma)
        mmd_values.append(mmd2)

    mmd_values = np.array(mmd_values)

    return {
        'mean': float(np.mean(mmd_values)),
        'std': float(np.std(mmd_values)),
        'min': float(np.min(mmd_values)),
        'max': float(np.max(mmd_values)),
        'p25': float(np.percentile(mmd_values, 25)),
        'median': float(np.median(mmd_values)),
        'p75': float(np.percentile(mmd_values, 75)),
        'sigma_mean': float(np.mean(sigmas)),
        'sigma_std': float(np.std(sigmas)),
    }


def compute_mmd_matrix(
    embeddings_dict: dict[str, np.ndarray],
    sample_size: int,
    num_repeats: int = 100,
    seed: int = 42
) -> tuple[np.ndarray, dict]:
    """Compute 3x3 MMD matrix for Real/Qwen/Llama.

    Args:
        embeddings_dict: dict mapping source to embeddings array
        sample_size: equal sample size for pairwise MMD
        num_repeats: number of bootstrap repeats
        seed: random seed

    Returns:
        mmd_matrix: 3x3 mean MMD matrix
        details: dict with full statistics for each pair
    """
    sources = ['real', 'qwen', 'llama']
    mmd_matrix = np.zeros((3, 3))
    details = {}

    for i, src_a in enumerate(sources):
        for j, src_b in enumerate(sources):
            if i == j:
                # Diagonal: MMD(A, A) = 0
                mmd_matrix[i, j] = 0.0
            elif i < j:
                # Compute MMD
                X = embeddings_dict[src_a]
                Y = embeddings_dict[src_b]

                stats = compute_mmd_with_bootstrap(X, Y, sample_size, num_repeats, seed)

                mmd_matrix[i, j] = stats['mean']
                mmd_matrix[j, i] = stats['mean']  # Symmetric

                details[f"{src_a}_vs_{src_b}"] = stats

    return mmd_matrix, details


def compute_class_level_mmd(
    embeddings_data: dict[str, dict],
    label: int,
    sample_size: int,
    num_repeats: int = 100,
    seed: int = 42
) -> tuple[np.ndarray, dict]:
    """Compute MMD for a specific class.

    Args:
        embeddings_data: dict with 'embeddings', 'labels' for each source
        label: class label (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
        sample_size: equal sample size
        num_repeats: bootstrap repeats
        seed: random seed

    Returns:
        mmd_matrix: 3x3 mean MMD matrix for this class
        details: statistics for each pair
    """
    sources = ['real', 'qwen', 'llama']

    # Filter by class
    class_embeddings = {}
    for source in sources:
        labels = np.array(embeddings_data[source]['labels'])
        embeddings = embeddings_data[source]['embeddings']
        mask = (labels == label)
        class_embeddings[source] = embeddings[mask]
        print(f"  {source.upper()} class {label}: {class_embeddings[source].shape[0]} samples")

    # Compute MMD matrix for this class
    mmd_matrix, details = compute_mmd_matrix(class_embeddings, sample_size, num_repeats, seed)

    return mmd_matrix, details


def compute_all_mmd_matrices(
    embeddings_data: dict[str, dict],
    sample_size_overall: int = 200,
    sample_size_class: int = 50,
    num_repeats: int = 100,
    seed: int = 42
) -> dict[str, Any]:
    """Compute all 5 MMD matrices (overall + 4 classes).

    Returns:
        dict with matrices and detailed statistics
    """
    print("="*60)
    print("MMD Analysis")
    print("="*60)

    results = {}

    # Overall MMD
    print("\n1. Overall MMD:")
    overall_embeddings = {
        src: data['embeddings'] for src, data in embeddings_data.items()
    }
    overall_matrix, overall_details = compute_mmd_matrix(
        overall_embeddings, sample_size_overall, num_repeats, seed
    )
    results['overall'] = {
        'matrix': overall_matrix,
        'details': overall_details,
    }

    # Class-level MMD
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    for label, name in label_names.items():
        print(f"\n{label+2}. {name} MMD:")
        class_matrix, class_details = compute_class_level_mmd(
            embeddings_data, label, sample_size_class, num_repeats, seed
        )
        # Use safe key name without special characters
        safe_name = name.lower().replace('/', '_').replace(' ', '_')
        results[safe_name] = {
            'matrix': class_matrix,
            'details': class_details,
        }

    return results


def create_mmd_dataframes(mmd_results: dict) -> dict[str, pd.DataFrame]:
    """Create DataFrames for MMD matrices."""
    sources = ['Real', 'Qwen', 'Llama']

    dfs = {}
    for scope, data in mmd_results.items():
        df = pd.DataFrame(data['matrix'], index=sources, columns=sources)
        dfs[scope] = df

    return dfs


if __name__ == "__main__":
    from load_phase1_data import load_all_data
    from compute_embeddings import compute_and_cache_all_embeddings

    # Load data
    all_data = load_all_data()

    # Compute/load embeddings
    embeddings_data = compute_and_cache_all_embeddings(all_data)

    # Compute MMD
    mmd_results = compute_all_mmd_matrices(embeddings_data)

    # Create DataFrames
    dfs = create_mmd_dataframes(mmd_results)

    print("\n" + "="*60)
    print("MMD Matrices")
    print("="*60)

    for scope, df in dfs.items():
        print(f"\n{scope.upper()} MMD:")
        print(df)

    # Save
    output_dir = Path("outputs/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    for scope, df in dfs.items():
        df.to_csv(output_dir / f"mmd_{scope}.csv")

    print(f"\nSaved to {output_dir}/")
