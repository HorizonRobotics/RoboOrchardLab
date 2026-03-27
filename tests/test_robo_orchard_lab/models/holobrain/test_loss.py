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

import pytest
import torch

from robo_orchard_lab.models.holobrain.loss import HoloBrainActionLoss


def _make_model_outs(
    bs=2,
    num_steps=16,
    num_joints=7,
    state_dims=8,
    num_parallel=None,
    timesteps=None,
):
    """Create minimal model_outs dict for testing."""
    total_bs = bs * num_parallel if num_parallel else bs
    pred = torch.randn(total_bs, num_steps, num_joints, state_dims)
    target = torch.randn(total_bs, num_steps, num_joints, state_dims)
    if timesteps is None:
        timesteps = torch.randint(0, 1000, (total_bs,))
    return {
        "pred": pred,
        "target": target,
        "timesteps": timesteps,
        "num_parallel": num_parallel,
    }


class TestHoloBrainActionLossBasic:
    """Basic loss mode tests."""

    @pytest.mark.parametrize("loss_mode", ["l1", "l2", "smooth_l1"])
    def test_loss_modes_return_scalar(self, loss_mode):
        """Each loss mode returns a dict of scalar losses."""
        loss_fn = HoloBrainActionLoss(loss_mode=loss_mode)
        model_outs = _make_model_outs()
        inputs = {}
        result = loss_fn(model_outs, inputs)

        assert "loss_angle" in result
        assert "loss_xyz" in result
        assert "loss_rot" in result
        for key, val in result.items():
            assert val.shape == (), f"{key} should be scalar, got {val.shape}"

    def test_l2_loss_positive_for_random_inputs(self):
        """L2 loss is non-negative."""
        loss_fn = HoloBrainActionLoss(loss_mode="l2")
        model_outs = _make_model_outs()
        result = loss_fn(model_outs, {})
        assert result["loss_angle"] >= 0
        assert result["loss_xyz"] >= 0
        assert result["loss_rot"] >= 0

    def test_zero_loss_for_identical_pred_target(self):
        """Loss is zero when prediction equals target."""
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=False
        )
        bs, num_steps, num_joints, state_dims = 2, 16, 7, 8
        pred = torch.randn(bs, num_steps, num_joints, state_dims)
        model_outs = {
            "pred": pred.clone(),
            "target": pred.clone(),
            "timesteps": torch.zeros(bs, dtype=torch.long),
            "num_parallel": None,
        }
        result = loss_fn(model_outs, {})
        assert torch.allclose(
            result["loss_angle"], torch.tensor(0.0), atol=1e-6
        )
        assert torch.allclose(result["loss_xyz"], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(result["loss_rot"], torch.tensor(0.0), atol=1e-6)

    def test_invalid_loss_mode_raises(self):
        """Invalid loss mode raises AssertionError."""
        with pytest.raises(AssertionError):
            HoloBrainActionLoss(loss_mode="huber")


class TestHoloBrainActionLossWeight:
    """Tests for loss weighting options."""

    def test_with_state_loss_weight(self):
        """State loss weight scales the loss values."""
        bs, num_steps, num_joints, state_dims = 2, 16, 7, 8
        # weight has same last dim as state_dims
        weight = torch.ones(bs, num_steps, num_joints, state_dims)
        loss_fn_w = HoloBrainActionLoss(
            loss_mode="l2",
            default_state_loss_weight=weight,
            with_wasserstein_distance=False,
        )
        loss_fn_no_w = HoloBrainActionLoss(
            loss_mode="l2",
            with_wasserstein_distance=False,
        )
        model_outs = _make_model_outs(
            bs=bs, num_steps=num_steps, num_joints=num_joints
        )
        r_w = loss_fn_w(model_outs, {})
        r_no_w = loss_fn_no_w(model_outs, {})
        # With all-ones weight, losses should be the same
        assert torch.allclose(
            r_w["loss_angle"], r_no_w["loss_angle"], atol=1e-5
        )

    def test_zero_weight_gives_zero_loss(self):
        """Zero state loss weight produces zero loss."""
        bs, num_steps, num_joints, state_dims = 2, 16, 7, 8
        weight = torch.zeros(bs, num_steps, num_joints, state_dims)
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2",
            with_wasserstein_distance=False,
        )
        model_outs = _make_model_outs(
            bs=bs, num_steps=num_steps, num_joints=num_joints
        )
        result = loss_fn(model_outs, {"state_loss_weights": weight})
        assert torch.allclose(
            result["loss_angle"], torch.tensor(0.0), atol=1e-6
        )
        assert torch.allclose(result["loss_xyz"], torch.tensor(0.0), atol=1e-6)

    def test_timestep_loss_weight(self):
        """Timestep loss weight modulates loss by 1/(timestep+1)."""
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2",
            timestep_loss_weight=1000,
            with_wasserstein_distance=False,
        )
        model_outs = _make_model_outs()
        result = loss_fn(model_outs, {})
        # Should return scalar losses without error
        assert result["loss_angle"].shape == ()
        assert result["loss_xyz"].shape == ()


class TestHoloBrainActionLossMask:
    """Tests for prediction mask behavior."""

    def test_pred_mask_selects_subset(self):
        """pred_mask selects a subset of predictions for loss computation."""
        bs, num_steps, num_joints = 4, 16, 7
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=False
        )
        model_outs = _make_model_outs(
            bs=bs, num_steps=num_steps, num_joints=num_joints
        )
        # Only half the samples are valid
        pred_mask = torch.zeros(bs, dtype=torch.bool)
        pred_mask[:2] = True
        result = loss_fn(model_outs, {"pred_mask": pred_mask})
        assert result["loss_angle"].shape == ()

    def test_empty_pred_mask_returns_zero_loss(self):
        """All-False pred_mask (no valid samples) returns zero loss."""
        bs, num_steps, num_joints = 4, 16, 7
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=False
        )
        model_outs = _make_model_outs(
            bs=bs, num_steps=num_steps, num_joints=num_joints
        )
        pred_mask = torch.zeros(bs, dtype=torch.bool)  # all False
        result = loss_fn(model_outs, {"pred_mask": pred_mask})
        assert torch.allclose(
            result["loss_angle"], torch.tensor(0.0), atol=1e-6
        )


class TestHoloBrainActionLossParallel:
    """Tests for parallel trajectory (best-of-N) loss."""

    def test_parallel_samples_no_weight(self):
        """With num_parallel, averages over trajectories."""
        bs, num_parallel = 2, 4
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=False
        )
        model_outs = _make_model_outs(bs=bs, num_parallel=num_parallel)
        result = loss_fn(model_outs, {})
        assert result["loss_angle"].shape == ()

    def test_parallel_samples_with_weight(self):
        """With parallel_loss_weight, best trajectory is upweighted."""
        bs, num_parallel = 2, 4
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2",
            parallel_loss_weight=0.1,
            with_wasserstein_distance=False,
        )
        model_outs = _make_model_outs(bs=bs, num_parallel=num_parallel)
        result = loss_fn(model_outs, {})
        assert result["loss_angle"].shape == ()
        assert result["loss_angle"] >= 0


class TestHoloBrainActionLossWasserstein:
    """Tests for Wasserstein (rotation matrix) distance mode."""

    def test_with_wasserstein_distance(self):
        """with_wasserstein_distance=True converts quaternion to matrix."""
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=True
        )
        model_outs = _make_model_outs(state_dims=8)
        result = loss_fn(model_outs, {})
        assert "loss_rot" in result
        assert result["loss_rot"].shape == ()

    def test_without_wasserstein_distance(self):
        """with_wasserstein_distance=False uses raw quaternion loss."""
        loss_fn = HoloBrainActionLoss(
            loss_mode="l2", with_wasserstein_distance=False
        )
        model_outs = _make_model_outs(state_dims=8)
        result = loss_fn(model_outs, {})
        assert "loss_rot" in result
        assert result["loss_rot"].shape == ()


class TestHoloBrainActionLossFakeLoss:
    """Tests for the _fake_loss fallback."""

    def test_fake_loss_is_zero_scalar(self):
        """_fake_loss returns a zero scalar."""
        loss_fn = HoloBrainActionLoss(loss_mode="l2")
        error = torch.randn(4, 8)
        result = loss_fn._fake_loss(error)
        assert result.shape == ()
        assert result.item() == 0.0

    def test_fake_loss_preserves_gradient_connection(self):
        """_fake_loss output is connected to the graph (grad flows)."""
        loss_fn = HoloBrainActionLoss(loss_mode="l2")
        error = torch.randn(4, 8, requires_grad=True)
        result = loss_fn._fake_loss(error)
        result.backward()
        assert error.grad is not None
