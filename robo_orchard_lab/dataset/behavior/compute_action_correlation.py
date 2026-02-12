# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import math
import os

import numpy as np

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb


def compute_action_correlation_from_chunks(
    all_corr_chunks: np.ndarray,
    *,
    max_correlation_samples: int = 200_000,
    epsilon: float = 1e-6,
    constant_std_threshold: float = 1e-6,
    diagonal_zero_threshold: float = 1e-10,
    stronger_regularization_floor: float = 1e-5,
):
    """Compute the action correlation matrix.

    copy from: https://github.com/IliaLarchenko/behavior-1k-solution/tree/main.

    This computes:
      - full correlation matrix Cholesky (L) for flattened (H*D) dimensions
      - averaged spatial correlation (D x D), averaged over timesteps
      - averaged temporal correlation (H x H), averaged over non-constant dims

    Args:
        all_corr_chunks:
            np.ndarray, shape (N, H, D). (delta) action chunks.
        max_correlation_samples:
            number of samples used for correlation computation.
        epsilon:
            base diagonal regularization added before Cholesky.
        constant_std_threshold:
            threshold to treat dimensions as constant/padded (std ~ 0).
        diagonal_zero_threshold:
            threshold to treat diagonal as zero variance after cov.
        stronger_regularization_floor:
            minimum epsilon when strengthening regularization.

    Returns:
        correlation_stats:
            action_correlation_cholesky, (H*D, H*D)
            action_correlation_spatial, (D, D)
            action_correlation_temporal, (H, H)
        debug:
            dict of intermediate diagnostics (optional but useful)
    """
    num_chunk, action_horizon, action_dim = all_corr_chunks.shape

    # Flatten chunks to (num_samples, H * D)
    flattened_chunks = all_corr_chunks.reshape(
        num_chunk, action_horizon * action_dim
    )
    total_samples = flattened_chunks.shape[0]

    # Subsample if too many samples (same logic)
    if total_samples > max_correlation_samples:
        print(
            f"Subsampling {max_correlation_samples} chunks "
            f"from {total_samples}"
        )
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(
            total_samples, size=max_correlation_samples, replace=False
        )
        flattened_chunks = flattened_chunks[sample_indices]

    # Normalize each dimension to 0 mean / 1 std
    chunk_mean = np.mean(flattened_chunks, axis=0)
    chunk_std = np.std(flattened_chunks, axis=0)

    constant_dims = chunk_std < constant_std_threshold
    print(
        f"Found {np.sum(constant_dims)} constant dimensions, "
        f"will be set to identity"
    )

    # Normalize non-constant dimensions
    normalized_chunks = flattened_chunks.copy()
    normalized_chunks[:, ~constant_dims] = (
        flattened_chunks[:, ~constant_dims] - chunk_mean[~constant_dims]
    ) / chunk_std[~constant_dims]

    # Empirical covariance of normalized data
    # For normalized data, covariance == correlation (up to numerical noise)
    cov_matrix = np.cov(normalized_chunks, rowvar=False)

    # Enforce diagonal normalization and handle constant dims carefully
    diag_vals = np.diag(cov_matrix).copy()
    print(
        f"Diagonal before correction: "
        f"min={np.min(diag_vals):.6f}, max={np.max(diag_vals):.6f}"
    )

    constant_mask = diag_vals < diagonal_zero_threshold
    num_constant = np.sum(constant_mask)
    print(f"Found {num_constant} dimensions with zero variance on diagonal")

    diag_vals_safe = diag_vals.copy()
    diag_vals_safe[constant_mask] = 1.0  # Prevent division by zero

    normalizer = np.sqrt(diag_vals_safe[:, None] @ diag_vals_safe[None, :])
    cov_matrix = cov_matrix / normalizer

    # For constant dimensions: zero-out rows/cols, diagonal to 1
    cov_matrix[constant_mask, :] = 0.0
    cov_matrix[:, constant_mask] = 0.0
    np.fill_diagonal(cov_matrix, 1.0)

    print(
        f"Diagonal after correction: "
        f"min={np.min(np.diag(cov_matrix)):.6f},"
        f"max={np.max(np.diag(cov_matrix)):.6f}"
    )
    print(
        f"Matrix check: "
        f"has NaN={np.any(np.isnan(cov_matrix))},"
        f"has Inf={np.any(np.isinf(cov_matrix))}"
    )

    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        raise ValueError(
            "Correlation matrix contains NaN/Inf after normalization"
        )

    # Add regularization for numerical stability
    cov_matrix_reg = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

    # Check eigenvalues for positive definiteness
    eigenvalues = np.linalg.eigvalsh(cov_matrix_reg)
    min_eigenvalue = float(np.min(eigenvalues))
    print(f"Min eigenvalue after regularization: {min_eigenvalue:.6e}")

    if min_eigenvalue <= 0:
        print(
            "WARNING: Matrix not positive definite!"
            "Adding stronger regularization..."
        )
        eps2 = max(
            stronger_regularization_floor,
            -min_eigenvalue + stronger_regularization_floor,
        )
        cov_matrix_reg = cov_matrix + eps2 * np.eye(cov_matrix.shape[0])
        eigenvalues2 = np.linalg.eigvalsh(cov_matrix_reg)
        min_eigenvalue2 = float(np.min(eigenvalues2))
        print(f"New min eigenvalue: {min_eigenvalue2:.6e}")

        # Keep for debug
        min_eigenvalue = min_eigenvalue2
        epsilon_used = eps2
    else:
        epsilon_used = epsilon

    # Cholesky decomposition
    try:
        chol_lower = np.linalg.cholesky(cov_matrix_reg)
        print(f"Cholesky decomposition successful! Shape: {chol_lower.shape}")

        reconstructed = chol_lower @ chol_lower.T
        reconstruction_error = np.linalg.norm(
            reconstructed - cov_matrix_reg, "fro"
        ) / np.linalg.norm(cov_matrix_reg, "fro")
        print(f"Cholesky reconstruction error: {reconstruction_error:.6e}")

    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Cholesky decomposition failed: {e}. "
            "This indicates the covariance matrix is not positive definite "
            "even after regularization. "
            "This is a critical error that needs investigation."
        )

    # Averaged spatial correlation (D x D)
    # Average correlation across all timesteps
    print("\nComputing averaged spatial correlation (dim × dim)...")

    spatial_corrs = []
    for t in range(action_horizon):
        start_idx = t * action_dim
        end_idx = (t + 1) * action_dim
        # (num_samples_used, D)
        timestep_data = normalized_chunks[:, start_idx:end_idx]

        corr_t = np.cov(timestep_data, rowvar=False)  # (D, D)

        # Normalize to ensure diagonal = 1
        diag_t = np.diag(corr_t)
        # Guard against zeros on diagonal
        # rare but possible if a dim constant at this timestep
        diag_t_safe = diag_t.copy()
        diag_t_safe[diag_t_safe < diagonal_zero_threshold] = 1.0
        corr_t = corr_t / np.sqrt(diag_t_safe[:, None] @ diag_t_safe[None, :])

        spatial_corrs.append(corr_t)

    avg_spatial_corr = np.mean(spatial_corrs, axis=0)
    np.fill_diagonal(avg_spatial_corr, 1.0)

    # spatial sanity checks
    print(f"Averaged spatial correlation shape: {avg_spatial_corr.shape}")

    print(
        f"Spatial check: symmetric="
        f"{np.allclose(avg_spatial_corr, avg_spatial_corr.T, atol=1e-6)} "
        f"diag_min={np.min(np.diag(avg_spatial_corr)):.6f} "
        f"diag_max={np.max(np.diag(avg_spatial_corr)):.6f} "
        f"hasNaN={np.any(np.isnan(avg_spatial_corr))} "
        f"hasInf={np.any(np.isinf(avg_spatial_corr))}"
    )

    # Averaged temporal correlation (H x H)
    # Average correlation across all NON-CONSTANT dimensions only
    print("Computing averaged temporal correlation (time × time)...")

    temporal_corrs = []

    # Identify which original dimensions are constant across all timesteps
    dim_is_constant = []
    for d in range(action_dim):
        dim_indices_check = [t * action_dim + d for t in range(action_horizon)]
        # note: chunk_std computed on flattened_chunks
        dim_std = chunk_std[dim_indices_check]
        dim_is_constant.append(np.all(dim_std < constant_std_threshold))

    num_constant_dims = int(np.sum(dim_is_constant))
    print(
        f"Excluding {num_constant_dims} constant dimensions "
        f"from temporal averaging"
    )

    for d in range(action_dim):
        if dim_is_constant[d]:
            continue

        dim_indices = [t * action_dim + d for t in range(action_horizon)]
        dim_data = normalized_chunks[:, dim_indices]  # (num_samples_used, H)

        corr_d = np.cov(dim_data, rowvar=False)  # (H, H)

        diag_d = np.diag(corr_d)
        diag_d_safe = diag_d.copy()
        diag_d_safe[diag_d_safe < diagonal_zero_threshold] = 1.0
        corr_d = corr_d / np.sqrt(diag_d_safe[:, None] @ diag_d_safe[None, :])

        temporal_corrs.append(corr_d)

    if len(temporal_corrs) > 0:
        avg_temporal_corr = np.mean(temporal_corrs, axis=0)
        np.fill_diagonal(avg_temporal_corr, 1.0)
        print(
            f"Averaged temporal correlation shape: {avg_temporal_corr.shape}"
        )
    else:
        avg_temporal_corr = np.eye(action_horizon)
        print(
            "All dimensions constant - using identity for temporal correlation"
        )

    print(
        f"Temporal check: symmetric="
        f"{np.allclose(avg_temporal_corr, avg_temporal_corr.T, atol=1e-6)} "
        f"diag_min={np.min(np.diag(avg_temporal_corr)):.6f} "
        f"diag_max={np.max(np.diag(avg_temporal_corr)):.6f} "
        f"hasNaN={np.any(np.isnan(avg_temporal_corr))} "
        f"hasInf={np.any(np.isinf(avg_temporal_corr))}"
    )

    # Store outputs (same keys as your pipeline)
    correlation_stats = {
        "action_correlation_cholesky": chol_lower,
        "action_correlation_spatial": avg_spatial_corr,
        "action_correlation_temporal": avg_temporal_corr,
    }

    debug = {
        "N_original": num_chunk,
        "N_used": normalized_chunks.shape[0],
        "H": action_horizon,
        "D": action_dim,
        "epsilon_used": float(epsilon_used),
        "min_eigenvalue": float(min_eigenvalue),
        "num_constant_dims_flat": int(np.sum(constant_dims)),
        "num_constant_diag": int(num_constant),
        "reconstruction_error": float(reconstruction_error),
    }

    return correlation_stats, debug


if __name__ == "__main__":
    action_horizon = 64
    all_action_chunks = []

    root_dir = "/work/bucket/dataset/behavior1k_lmdb_data_v5/"
    for task in os.listdir(root_dir):
        print(task)
        input_path = os.path.join(root_dir, task)
        index_lmdb = Lmdb(f"{input_path}/index/", writable=False)
        meta_lmdb = Lmdb(f"{input_path}/meta/", writable=False)

        keys = sorted([int(i) for i in index_lmdb.keys()])
        for key in keys:
            idx = index_lmdb.get(key)
            uuid = idx["uuid"]
            num_steps = idx["num_steps"]

            num_steps_per_shard = meta_lmdb[f"{uuid}/num_steps_per_shard"]
            if num_steps_per_shard is not None:
                action = []
                num_shards = math.ceil(num_steps / num_steps_per_shard)
                for i in range(num_shards):
                    act = meta_lmdb[f"{uuid}/{i}/robot_action/joint_position"]
                    if act is not None:
                        action.append(act)
                action = np.concatenate(action, axis=0)
            else:
                action = meta_lmdb[f"{uuid}/robot_action/joint_position"]

            n, action_dim = action.shape
            num_chunks = n // action_horizon
            if num_chunks == 0:
                continue

            action = action[: num_chunks * action_horizon]
            action_chunks = action.reshape(
                num_chunks, action_horizon, action_dim
            )

            all_action_chunks.append(action_chunks)

    all_action_chunks = np.concatenate(all_action_chunks, axis=0)
    np.save("action_chunk.npy", all_action_chunks)
    print(all_action_chunks.shape)

    all_action_chunks = np.load("action_chunk.npy")
    all_action_chunks = all_action_chunks[:, :, 3:]
    print(all_action_chunks.shape)
    corr_stats, _ = compute_action_correlation_from_chunks(all_action_chunks)
    np.save("action_corr_stats.npy", corr_stats["action_correlation_cholesky"])
