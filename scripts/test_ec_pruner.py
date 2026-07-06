import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "llava" / "model" / "pruners" / "ec_pruner.py"
CLIP_ENCODER_PATH = REPO_ROOT / "llava" / "model" / "multimodal_encoder" / "clip_encoder.py"


def load_ec_pruner_class():
    spec = importlib.util.spec_from_file_location("ec_pruner_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ECPruner


def load_budget_calibrate_prob():
    spec = importlib.util.spec_from_file_location("ec_pruner_budget_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.budget_calibrate_prob


def load_projection_overlap_kernel():
    spec = importlib.util.spec_from_file_location("ec_pruner_overlap_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_projection_overlap_kernel


def load_spectral_filter_kernel():
    spec = importlib.util.spec_from_file_location("ec_pruner_spectral_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.spectral_filter_kernel


def load_select_by_evidence_recovery():
    spec = importlib.util.spec_from_file_location("ec_pruner_evidence_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_by_evidence_recovery


def load_select_by_core_then_recover():
    spec = importlib.util.spec_from_file_location("ec_pruner_core_recover_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_by_core_then_recover


def load_adaptive_recover_helpers():
    spec = importlib.util.spec_from_file_location("ec_pruner_adaptive_recover_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.infer_question_recovery_type, module.select_by_adaptive_core_recover


def load_select_by_qmo_cr():
    spec = importlib.util.spec_from_file_location("ec_pruner_qmo_cr_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_by_qmo_cr


def load_qmo_cr_helpers():
    spec = importlib.util.spec_from_file_location("ec_pruner_qmo_cr_helpers_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.infer_qmo_recovery_type, module.select_by_qmo_cr


def load_qmo_gated_helpers():
    spec = importlib.util.spec_from_file_location("ec_pruner_qmo_gated_helpers_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.infer_qmo_gate_type, module.select_by_qmo_gated


def load_select_by_qsp_density_projective():
    spec = importlib.util.spec_from_file_location("ec_pruner_qsp_density_projective_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_by_qsp_density_projective


def load_qfi_cards_helpers():
    spec = importlib.util.spec_from_file_location("ec_pruner_qfi_cards_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.select_by_qfi_zeno_stabilized,
        module.select_by_qfi_spatial_buffer_recovery,
        module.select_by_qfi_progressive,
        module.select_by_qfi_coarse_to_fine,
    )


def load_clip_vision_tower_class():
    spec = importlib.util.spec_from_file_location("clip_encoder_module", CLIP_ENCODER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIPVisionTower


@contextmanager
def env_override(**kwargs):
    backup = {key: os.environ.get(key) for key in kwargs}
    try:
        for key, value in kwargs.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device):
    assert keep_idx.ndim == 1
    assert keep_idx.numel() == keep_num
    assert keep_idx.device.type == device.type
    assert torch.unique(keep_idx).numel() == keep_num
    assert int(keep_idx.min().item()) >= 0
    assert int(keep_idx.max().item()) < num_tokens
    assert torch.equal(keep_idx, keep_idx.sort().values)


def assert_finite_debug_values(info):
    for key, value in info.items():
        if isinstance(value, float):
            assert value == value
            assert value != float("inf")
            assert value != float("-inf")


def make_psd_kernel(size, device, dtype=torch.float32, rank=None):
    rank = size if rank is None else max(1, min(int(rank), int(size)))
    base = torch.randn(size, rank, device=device, dtype=dtype)
    kernel = base @ base.t()
    kernel = 0.5 * (kernel + kernel.t())
    return kernel


def test_greedy_solver_fixed_k(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="norm",
        EC_USE_CHAIN_COVERAGE="0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 1024, device=device)
        keep_idx, _ = pruner.select(visual_tokens, keep_num=64, question="what is on the table")
        assert_valid_keep_idx(keep_idx, 64, 576, device)


def test_qanneal_solver_fixed_k(ECPruner, device):
    with env_override(
        EC_SOLVER="qanneal",
        EC_QA_STEPS="20",
        EC_QA_T0="1.0",
        EC_QA_TEND="0.01",
        EC_QA_SEED="42",
        EC_QA_MAX_SWAP_RATIO="0.15",
        EC_SCORE_SOURCE="norm",
        EC_USE_CHAIN_COVERAGE="0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 1024, device=device)
        keep_idx, debug_info = pruner.select(visual_tokens, keep_num=64, question="what is on the table")
        solver_debug = debug_info["solver_debug"]
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert solver_debug["solver_used"] == "qanneal"
        assert solver_debug["best_energy"] <= solver_debug["greedy_energy"] + 1e-6


def test_candidate_padding_to_k(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_GLOBAL_CANDIDATE_RATIO="0.125",
        EC_CANDIDATE_RATIO="1.0",
        EC_USE_CHAIN_COVERAGE="0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
    ):
        pruner = ECPruner(debug=False)
        a = torch.linspace(0.0, 1.0, steps=80, device=device)
        candidate_idx, meta = pruner.build_semantic_candidate_pool(a, None, keep_num=64)
        assert candidate_idx.ndim == 1
        assert candidate_idx.numel() == 64
        assert meta["candidate_padded_count"] > 0


def test_env_bool_parsing(ECPruner):
    with env_override(
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="1",
        EC_SPATIAL_EXCLUDE_GLOBAL="0",
        EC_SOLVER="qanneal",
    ):
        pruner = ECPruner(debug=False)
        assert pruner.use_repulsion is False
        assert pruner.use_complement is False
        assert pruner.use_phi is False
        assert pruner.use_semantic_candidate is False
        assert pruner.use_spatial_candidate is True
        assert pruner.spatial_exclude_global is False
        assert pruner.solver == "qanneal"


def test_projection_overlap_kernel_modes(device):
    compute_kernel = load_projection_overlap_kernel()
    tokens = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        device=device,
    )
    relu_square = compute_kernel(tokens, overlap_kernel="relu_square")
    abs_square = compute_kernel(tokens, overlap_kernel="abs_square")
    cosine = compute_kernel(tokens, overlap_kernel="cosine")

    assert relu_square.shape == (3, 3)
    assert abs_square.shape == (3, 3)
    assert cosine.shape == (3, 3)
    assert torch.all((relu_square >= 0.0) & (relu_square <= 1.0))
    assert torch.all((abs_square >= 0.0) & (abs_square <= 1.0))
    assert torch.all((cosine >= 0.0) & (cosine <= 1.0))
    assert float(relu_square[0, 1].item()) == 0.0
    assert float(abs_square[0, 1].item()) > 0.99
    assert float(cosine[0, 1].item()) == 0.0


def test_pairwise_matrices_are_symmetric(ECPruner, device):
    with env_override(
        EC_USE_REPULSION="1",
        EC_USE_COMPLEMENT="1",
        EC_USE_PHI="1",
        EC_USE_CHAIN_COVERAGE="0",
        EC_TOPK_R="4",
        EC_TOPK_C="4",
    ):
        pruner = ECPruner(debug=False)
        candidate_size = 24
        feat_dim = 32
        unit_count = 3
        features_c = torch.randn(candidate_size, feat_dim, device=device)
        b_c = torch.softmax(torch.randn(candidate_size, unit_count, device=device), dim=-1)
        positions_c = pruner.build_patch_positions(candidate_size, device)
        a_c = torch.rand(candidate_size, device=device)
        relation_matrix = pruner.build_relation_matrix(["left", "table", "holding"], device=device)

        repulsion = pruner.compute_repulsion(features_c, b_c, positions_c)
        complementarity = pruner.compute_complementarity(
            a_c,
            b_c,
            relation_matrix,
            positions_c,
            "what is on the left of the table",
        )

        assert torch.allclose(repulsion, repulsion.t(), atol=1e-6)
        assert torch.allclose(complementarity, complementarity.t(), atol=1e-6)
        assert torch.allclose(torch.diag(repulsion), torch.zeros(candidate_size, device=device), atol=1e-6)
        assert torch.allclose(
            torch.diag(complementarity),
            torch.zeros(candidate_size, device=device),
            atol=1e-6,
        )


def test_qmo_score_and_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qmo",
        EC_W_REL="0.40",
        EC_W_SEM="0.25",
        EC_W_SPATIAL="0.15",
        EC_W_CHAIN="0.15",
        EC_W_ENT="0.05",
        EC_QMO_INIT_SOURCE="semantic",
        EC_USE_CHAIN_COVERAGE="0",
        EC_USE_SEMANTIC_CANDIDATE="1",
        EC_USE_SPATIAL_CANDIDATE="1",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        b = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        relation = pruner.build_relation_matrix(["left", "table", "red", "cup"], device=device)
        qmo_score = pruner.compute_node_score(
            visual_tokens,
            b=b,
            semantic_response_source="clip",
            relation_matrix=relation,
            K=64,
        )
        assert qmo_score.ndim == 1
        assert qmo_score.numel() == 576
        assert torch.isfinite(qmo_score).all()
        assert float(qmo_score.min().item()) >= 0.0
        assert float(qmo_score.max().item()) <= 1.0 + 1e-6

        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is the red cup on the left of the table",
            b=b,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["score_source"] == "qmo"
        assert debug_info["qmo_info"]["qmo_enabled"] is True
        assert debug_info["qmo_info"]["initial_topk_size"] == 64


def test_qfid_amplitude_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="uniform",
        EC_QFID_KERNEL="amplitude",
        EC_QFID_DEPOLARIZE="0.0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["score_source"] == "qfid"
        assert debug_info["qfid_bypassed_energy_pipeline"] is True
        assert debug_info["use_semantic_candidate"] is False
        assert debug_info["use_spatial_candidate"] is False
        assert debug_info["use_repulsion"] is False
        assert debug_info["use_complement"] is False
        assert debug_info["use_chain_coverage"] is False
        assert debug_info["qfid_info"]["enabled"] is True
        assert debug_info["qfid_info"]["kernel"] == "amplitude"
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_density_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["kernel"] == "density"
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_clsmix_probability_and_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.05",
        EC_QFID_CLS_ATTN_LAYER="last",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, 576, device=device)

        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=semantic_response,
            cls_attn=cls_attention,
            semantic_response_source="clip",
        )
        assert source_used == "clsmix"
        assert prob_info["cls_attn_available"] is True
        assert prob_info["cls_fallback"] is False
        assert prob_info["cls_mix_mode"] == "linear"
        assert abs(prob_info["cls_mix_beta"] - 0.05) <= 1e-8
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert prob_info["p_cls_max"] >= prob_info["p_cls_min"] >= 0.0
        assert prob_info["p_mix_max"] >= prob_info["p_mix_min"] >= 0.0

        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
            cls_attn=cls_attention,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["prob_source"] == "clsmix"
        assert debug_info["qfid_info"]["cls_attn_available"] is True
        assert debug_info["qfid_info"]["cls_fallback"] is False
        assert debug_info["qfid_info"]["cls_mix_mode"] == "linear"
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_clsmix_geometric_probability_and_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="geometric",
        EC_QFID_CLS_MIX_BETA="0.10",
        EC_QFID_CLS_ATTN_LAYER="last",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.0",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 576
        visual_tokens = torch.randn(1, num_tokens, 128, device=device)
        semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, num_tokens, device=device)

        probs, source_used, prob_info = pruner.compute_qfid_probability(
            num_tokens,
            device,
            b=semantic_response,
            cls_attn=cls_attention,
            semantic_response_source="clip",
        )
        semantic_score = pruner.compute_semantic_node_score(semantic_response)
        p_sem = pruner._softmax_probability_from_score(
            semantic_score,
            pruner.qfid_tau,
            device,
        )
        p_cls = cls_attention.reshape(-1).float()
        p_cls = p_cls / p_cls.sum().clamp_min(pruner.qfid_eps)
        beta = pruner.qfid_cls_mix_beta
        expected_log = (
            (1.0 - beta) * torch.log(p_sem + pruner.qfid_eps)
            + beta * torch.log(p_cls + pruner.qfid_eps)
        )
        expected = torch.exp(expected_log - expected_log.max())
        expected = expected / expected.sum().clamp_min(pruner.qfid_eps)

        assert source_used == "clsmix"
        assert prob_info["cls_mix_mode"] == "geometric"
        assert prob_info["cls_attn_available"] is True
        assert prob_info["cls_fallback"] is False
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs, expected, atol=1e-6, rtol=1e-5)
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5

        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
            cls_attn=cls_attention,
        )
        assert_valid_keep_idx(keep_idx, 64, num_tokens, device)
        assert debug_info["qfid_info"]["cls_mix_mode"] == "geometric"
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_clsmix_fallback_to_semantic(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.05",
        EC_QFID_CLS_ATTN_LAYER="last",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
    ):
        pruner = ECPruner(debug=False)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=semantic_response,
            cls_attn=None,
            semantic_response_source="clip",
        )
        assert source_used == "semantic_fallback"
        assert prob_info["cls_attn_available"] is False
        assert prob_info["cls_fallback"] is True
        assert prob_info["source_fallback"] is True
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5


def test_qfid_clsmix_gate_disabled_matches_legacy(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.105",
        EC_QFID_CLS_GATE="0",
        EC_QFID_CLS_GATE_BETA_BASE="0.90",
        EC_QFID_CLS_ATTN_LAYER="-2",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE="0.0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 96
        semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, num_tokens, device=device)
        probs, source_used, info = pruner.compute_qfid_probability(
            num_tokens,
            device,
            b=semantic_response,
            cls_attn=cls_attention,
            semantic_response_source="clip",
        )
        semantic_score = pruner.compute_semantic_node_score(semantic_response)
        p_sem = pruner._softmax_probability_from_score(semantic_score, pruner.qfid_tau, device)
        p_cls = cls_attention.reshape(-1).float()
        p_cls = p_cls / p_cls.sum().clamp_min(pruner.qfid_eps)
        expected = (1.0 - 0.105) * p_sem + 0.105 * p_cls
        expected = expected / expected.sum().clamp_min(pruner.qfid_eps)

        assert source_used == "clsmix"
        assert info["cls_gate_enabled"] is False
        assert abs(info["beta_eff"] - 0.105) <= 1e-8
        assert torch.allclose(probs, expected, atol=1e-7, rtol=1e-6)


def test_qfid_clsmix_agreement_gate_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.105",
        EC_QFID_CLS_GATE="1",
        EC_QFID_CLS_GATE_MODE="agreement",
        EC_QFID_CLS_GATE_BETA_BASE="0.105",
        EC_QFID_CLS_GATE_MIN="0.5",
        EC_QFID_CLS_GATE_MAX="1.5",
        EC_QFID_CLS_ATTN_LAYER="-2",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 576
        visual_tokens = torch.randn(1, num_tokens, 128, device=device)
        semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, num_tokens, device=device)
        probs, source_used, info = pruner.compute_qfid_probability(
            num_tokens,
            device,
            b=semantic_response,
            cls_attn=cls_attention,
            semantic_response_source="clip",
        )

        assert source_used == "clsmix"
        assert info["cls_gate_enabled"] is True
        assert info["cls_gate_mode"] == "agreement"
        assert 0.0 <= info["agreement"] <= 1.0
        assert 0.0525 - 1e-8 <= info["beta_eff"] <= 0.1575 + 1e-8
        assert abs(info["p_mix_sum"] - 1.0) <= 1e-5
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert info["p_sem_entropy"] >= 0.0
        assert info["p_cls_entropy"] >= 0.0

        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
            cls_attn=cls_attention,
        )
        assert_valid_keep_idx(keep_idx, 64, num_tokens, device)
        assert debug_info["qfid_info"]["keep_size"] == 64
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_cls_gate_does_not_affect_semantic_source(ECPruner, device):
    semantic_response = torch.softmax(torch.randn(64, 4, device=device), dim=-1)
    outputs = []
    for gate_enabled in ("0", "1"):
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_PROB_SOURCE="semantic",
            EC_QFID_CLS_GATE=gate_enabled,
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE="0.0",
        ):
            pruner = ECPruner(debug=False)
            probs, source_used, _ = pruner.compute_qfid_probability(
                64,
                device,
                b=semantic_response,
                semantic_response_source="clip",
            )
            assert source_used == "semantic"
            outputs.append(probs)
    assert torch.equal(outputs[0], outputs[1])


def test_qfid_cls_only_qf_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="qf",
        EC_QFID_PROB_SOURCE="cls_only_qf",
        EC_QFID_CLS_ATTN_LAYER="last",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 64, device=device)
        cls_attention = torch.rand(1, 4, 576, device=device)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            cls_attn=cls_attention,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["solver_used"] == "qfid_residual_greedy"
        assert debug_info["qfid_info"]["select_mode"] == "qf"
        assert debug_info["qfid_info"]["prob_source_used"] == "cls_only_qf"
        assert debug_info["qfid_info"]["cls_attn_available"] is True
        assert debug_info["qfid_info"]["cls_fallback"] is False
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_cls_topk_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="cls_topk",
        EC_QFID_PROB_SOURCE="cls_only_qf",
        EC_QFID_CLS_ATTN_LAYER="last",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 80
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        cls_attention = torch.arange(num_tokens, dtype=torch.float32, device=device).unsqueeze(0)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=16,
            cls_attn=cls_attention,
        )
        expected = torch.arange(num_tokens - 16, num_tokens, device=device)
        assert_valid_keep_idx(keep_idx, 16, num_tokens, device)
        assert torch.equal(keep_idx, expected)
        assert debug_info["selection_formulation"] == "cls_attention_ranking"
        assert debug_info["solver_used"] == "cls_topk"
        assert debug_info["qfid_info"]["select_mode"] == "cls_topk"


def test_qfid_semantic_topk_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="semantic_topk",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 80
        keep_num = 16
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        semantic_response = torch.zeros(num_tokens, 2, device=device)
        semantic_response[:, 0] = torch.linspace(0.50, 0.99, num_tokens, device=device)
        semantic_response[:, 1] = 1.0 - semantic_response[:, 0]
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=keep_num,
            b=semantic_response,
        )
        expected = torch.arange(num_tokens - keep_num, num_tokens, device=device)
        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert torch.equal(keep_idx, expected)
        assert debug_info["selection_formulation"] == "semantic_probability_ranking"
        assert debug_info["solver_used"] == "semantic_topk"
        assert debug_info["qfid_info"]["select_mode"] == "semantic_topk"


def test_qfid_uniform_density_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="qf",
        EC_QFID_PROB_SOURCE="uniform",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 96
        keep_num = 24
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        keep_idx, debug_info = pruner.select(visual_tokens, keep_num=keep_num)
        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert debug_info["solver_used"] == "qfid_residual_greedy"
        assert debug_info["qfid_info"]["kernel"] == "density"
        assert debug_info["qfid_info"]["prob_source_used"] == "uniform"
        assert debug_info["qfid_info"]["selector"] == "qfi_residual"


def test_evidence_recovery_selector_function(device):
    selector = load_select_by_evidence_recovery()
    for dtype in (torch.float32, torch.float16):
        states = torch.randn(80, 32, device=device, dtype=dtype)
        probs = torch.rand(80, device=device) * 10.0
        for keep_num in (32, 64, 128):
            keep_idx = selector(states, probs, keep_num)
            expected = min(keep_num, 80)
            assert keep_idx.ndim == 1
            assert keep_idx.numel() == expected
            assert keep_idx.device.type == device.type
            assert keep_idx.dtype == torch.long
            assert torch.unique(keep_idx).numel() == expected
            assert int(keep_idx.min().item()) >= 0
            assert int(keep_idx.max().item()) < 80
            assert torch.isfinite(keep_idx.float()).all()


def test_core_then_recover_selector_function(device):
    selector = load_select_by_core_then_recover()
    evidence_selector = load_select_by_evidence_recovery()
    for dtype in (torch.float32, torch.float16):
        states = torch.randn(80, 32, device=device, dtype=dtype)
        probs = torch.rand(80, device=device) * 10.0
        qfi_order = torch.randperm(80, device=device)
        for keep_num in (32, 64, 128):
            expected = min(keep_num, 80)
            for ratio in (0.5, 0.75):
                keep_idx = selector(states, probs, keep_num, qfi_order, core_ratio=ratio)
                assert keep_idx.ndim == 1
                assert keep_idx.numel() == expected
                assert keep_idx.device.type == device.type
                assert keep_idx.dtype == torch.long
                assert torch.unique(keep_idx).numel() == expected
                assert int(keep_idx.min().item()) >= 0
                assert int(keep_idx.max().item()) < 80
                assert torch.isfinite(keep_idx.float()).all()

        evidence_idx = evidence_selector(states, probs, 64)
        ratio0_idx = selector(states, probs, 64, qfi_order, core_ratio=0.0)
        assert torch.equal(ratio0_idx, evidence_idx)

        ratio1_idx = selector(states, probs, 64, qfi_order, core_ratio=1.0)
        assert torch.equal(ratio1_idx, qfi_order[:64].long())

        large_idx = selector(states, probs, 128, qfi_order, core_ratio=0.75)
        assert large_idx.numel() == 80
        assert torch.unique(large_idx).numel() == 80


def test_adaptive_core_recover_selector_function(device):
    infer_question_type, selector = load_adaptive_recover_helpers()
    assert infer_question_type("Is the car red?") == "binary"
    assert infer_question_type("What is next to the table?") == "relation"
    assert infer_question_type("Are the two objects the same color?") == "compare"
    assert infer_question_type("What color is the bus?") == "open"
    assert infer_question_type(None) == "default"

    common = dict(
        EC_QFID_ADAPT_MODE="entropy_question",
        EC_QFID_ADAPT_RECOVER_MIN_RATIO="0.00",
        EC_QFID_ADAPT_RECOVER_MAX_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO="0.125",
        EC_QFID_ADAPT_RECOVER_BINARY_RATIO="0.0625",
        EC_QFID_ADAPT_RECOVER_OPEN_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_REL_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_COMPARE_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_ATTR_RATIO="0.125",
        EC_QFID_ADAPT_RECOVER_CAP="16",
        EC_QFID_ADAPT_ENTROPY_GATE="0",
        EC_QFID_RECOVER_PRIOR_GAMMA="0.5",
    )
    for dtype in (torch.float32, torch.float16):
        states = torch.randn(80, 32, device=device, dtype=dtype)
        probs = torch.ones(80, device=device)
        qfi_order = torch.randperm(80, device=device)
        with env_override(**common, EC_QFID_RECOVER_PRIOR_ANCHOR="0", EC_QFID_RECOVER_CAND_POOL="0"):
            binary_idx, binary_info = selector(states, probs, 64, qfi_order, question_text="Is the car red?")
            open_idx, open_info = selector(states, probs, 64, qfi_order, question_text="What is next to the table?")
            none_idx, none_info = selector(states, probs, 64, qfi_order, question_text=None)
        assert binary_info["question_type"] == "binary"
        assert open_info["question_type"] == "relation"
        assert none_info["adapt_mode"] == "entropy_only"
        assert binary_info["k_recover"] < open_info["k_recover"]
        for keep_idx in (binary_idx, open_idx, none_idx):
            assert keep_idx.ndim == 1
            assert keep_idx.numel() == 64
            assert keep_idx.dtype == torch.long
            assert keep_idx.device.type == device.type
            assert torch.unique(keep_idx).numel() == 64
            assert int(keep_idx.min().item()) >= 0
            assert int(keep_idx.max().item()) < 80
            assert torch.isfinite(keep_idx.float()).all()

    states = torch.randn(96, 32, device=device)
    qfi_order = torch.randperm(96, device=device)
    low_entropy_probs = torch.zeros(96, device=device)
    low_entropy_probs[0] = 1.0
    high_entropy_probs = torch.ones(96, device=device)
    entropy_common = dict(common)
    entropy_common["EC_QFID_ADAPT_ENTROPY_GATE"] = "1"
    with env_override(**entropy_common, EC_QFID_RECOVER_PRIOR_ANCHOR="1", EC_QFID_RECOVER_CAND_POOL="1"):
        low_idx, low_info = selector(
            states, low_entropy_probs, 64, qfi_order, question_text="What is next to the table?"
        )
        high_idx, high_info = selector(
            states, high_entropy_probs, 64, qfi_order, question_text="What is next to the table?"
        )
        large_idx, large_info = selector(
            states, high_entropy_probs, 128, qfi_order, question_text="What is next to the table?"
        )
    assert low_info["k_recover"] <= high_info["k_recover"]
    assert high_info["k_recover"] <= 16
    assert high_info["recover_prior_anchor"] is True
    assert high_info["recover_cand_pool"] is True
    assert large_idx.numel() == 96
    assert torch.unique(large_idx).numel() == 96
    assert large_info["k_recover"] <= 16


def test_qficr_structural_ablation_controls(ECPruner, device):
    _, selector = load_adaptive_recover_helpers()
    states = torch.randn(64, 24, device=device)
    probs = torch.softmax(torch.randn(64, device=device), dim=0)
    qfi_order = torch.randperm(64, device=device)
    common = dict(
        EC_QFID_ADAPT_MODE="entropy_question",
        EC_QFID_ADAPT_RECOVER_MIN_RATIO="0.00",
        EC_QFID_ADAPT_RECOVER_MAX_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_REL_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_CAP="16",
        EC_QFID_ADAPT_ENTROPY_GATE="0",
        EC_QFID_RECOVER_PRIOR_ANCHOR="1",
        EC_QFID_RECOVER_PRIOR_GAMMA="0.5",
        EC_QFID_RECOVER_CAND_POOL="0",
    )
    with env_override(**common, EC_QFICR_RESTORATION_MODE="none"):
        keep_idx, info = selector(states, probs, 32, qfi_order, question_text="What is next to the table?")
        assert_valid_keep_idx(keep_idx.sort().values, 32, 64, device)
        assert info["restoration_mode"] == "none"
        assert info["k_recover"] == 0
        assert info["k_core"] == 32

    with env_override(**common, EC_QFICR_RESTORATION_MODE="fixed_ratio", EC_QFICR_FIXED_RESTORATION_RATIO="0.25"):
        keep_idx, info = selector(states, probs, 32, qfi_order, question_text="What is next to the table?")
        assert_valid_keep_idx(keep_idx.sort().values, 32, 64, device)
        assert info["restoration_mode"] == "fixed_ratio"
        assert info["k_recover"] == 8
        assert info["k_core"] == 24

    with env_override(**common, EC_QFICR_RESTORATION_MODE="random", EC_QFICR_RANDOM_SEED="123"):
        keep_a, info_a = selector(states, probs, 32, qfi_order, question_text="What is next to the table?")
    with env_override(**common, EC_QFICR_RESTORATION_MODE="random", EC_QFICR_RANDOM_SEED="123"):
        keep_b, info_b = selector(states, probs, 32, qfi_order, question_text="What is next to the table?")
    assert torch.equal(keep_a, keep_b)
    assert info_a["restoration_mode"] == "random"
    assert info_b["k_core"] + info_b["k_recover"] == 32
    assert torch.unique(keep_a).numel() == 32

    with env_override(
        **common,
        EC_QFICR_RESTORATION_MODE="full",
        EC_QFICR_USE_SPATIAL_LOCAL_PRIOR="1",
        EC_QFICR_PRIOR_ABLATION="no_spatial",
    ):
        keep_idx, info = selector(states, probs, 32, qfi_order, question_text="What is next to the table?")
        assert_valid_keep_idx(keep_idx.sort().values, 32, 64, device)
        assert info["use_spatial_local_prior"] is True
        assert info["prior_ablation"] == "no_spatial"

    qcommon = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_PROB_SOURCE": "clsmix",
        "EC_QFID_CLS_MIX_MODE": "linear",
        "EC_QFID_CLS_MIX_BETA": "0.105",
        "EC_QFID_CLS_GATE": "1",
        "EC_QFID_CLS_GATE_MODE": "agreement",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_DEPOLARIZE": "0.0",
    }
    semantic_response = torch.softmax(torch.randn(64, 4, device=device), dim=-1)
    cls_attention = torch.rand(1, 64, device=device)
    for mode in ("sem_only", "cls_only", "fixed_fusion"):
        with env_override(**qcommon, EC_QFICR_OBSERVATION_MODE=mode):
            pruner = ECPruner(debug=False)
            probs_out, source_used, prob_info = pruner.compute_qfid_probability(
                64,
                device,
                b=semantic_response,
                cls_attn=cls_attention,
                semantic_response_source="clip",
            )
            assert prob_info["observation_mode"] == mode
            assert source_used in {mode, "semantic_fallback", "cls_fallback", "uniform"}
            assert torch.isfinite(probs_out).all()
            assert abs(float(probs_out.sum().item()) - 1.0) <= 1e-5


def test_qmo_cr_selector_function(device):
    infer_qmo_type, selector = load_qmo_cr_helpers()
    assert infer_qmo_type("what color is the bus?") == "attr"
    assert infer_qmo_type("what text is written on the sign?") == "ocr"

    def check_case(states, probs, k, question_text, **env):
        with env_override(
            EC_QMO_USE_PURITY=env.get("EC_QMO_USE_PURITY", "1"),
            EC_QMO_W_COV=env.get("EC_QMO_W_COV", "1.0"),
            EC_QMO_W_H=env.get("EC_QMO_W_H", "0.5"),
            EC_QMO_LAMBDA_J=env.get("EC_QMO_LAMBDA_J", "0.2"),
            EC_QMO_RECOVER_CAP=env.get("EC_QMO_RECOVER_CAP", "16"),
            EC_QMO_LAMBDA_H=env.get("EC_QMO_LAMBDA_H", "0.5"),
            EC_QMO_LAMBDA_P=env.get("EC_QMO_LAMBDA_P", "0.5"),
        ):
            keep_idx, info = selector(states, probs, k, question_text=question_text)
        expected_k = min(k, states.shape[0])
        assert_valid_keep_idx(keep_idx.sort().values, expected_k, states.shape[0], device)
        assert info["qmo_cr_enabled"] is True
        assert info["k_core"] >= 1
        assert info["k_recover"] >= 0
        assert info["k_core"] + info["k_recover"] == expected_k
        assert 0.0 <= info["entropy"] <= 1.0
        assert 0.0 <= info["purity"] <= 1.0
        assert 0.0 <= info["D_q"] <= 1.0
        assert info["J_min"] >= 0.0
        assert info["J_max"] >= info["J_min"]
        assert_finite_debug_values({name: value for name, value in info.items() if isinstance(value, float)})
        return keep_idx, info

    random_states = torch.randn(576, 128, device=device)
    random_probs = torch.softmax(torch.randn(576, device=device), dim=0)
    for k in (32, 64, 128):
        keep_idx, info = check_case(random_states, random_probs, k, "what text is written on the sign")
        assert info["question_type"] == "ocr"

    same_states = torch.ones(96, 32, device=device) + 1e-7 * torch.randn(96, 32, device=device)
    equal_raw = torch.ones(96, device=device)
    _, same_info = check_case(same_states, equal_raw, 64, "what color is the bus")
    assert same_info["question_type"] == "attr"

    concentrated = torch.full((96,), 1e-9, device=device)
    concentrated[7] = 1.0
    check_case(torch.randn(96, 32, device=device), concentrated, 32, "is the car red")

    uniform = torch.ones(96, device=device)
    check_case(torch.randn(96, 32, device=device), uniform, 64, "what is next to the table")

    _, entropy_only = check_case(
        torch.randn(96, 32, device=device),
        torch.softmax(torch.randn(96, device=device), dim=0),
        32,
        "what object is shown",
        EC_QMO_USE_PURITY="0",
    )
    assert entropy_only["use_purity"] is False

    check_case(
        torch.randn(96, 32, device=device),
        torch.softmax(torch.randn(96, device=device), dim=0),
        32,
        "what object is shown",
        EC_QMO_LAMBDA_J="0.0",
    )
    check_case(
        torch.randn(96, 32, device=device),
        torch.softmax(torch.randn(96, device=device), dim=0),
        32,
        "what object is shown",
        EC_QMO_W_COV="0.0",
        EC_QMO_W_H="1.0",
        EC_QMO_LAMBDA_J="0.0",
    )

    _, unnormalized_info = check_case(
        torch.randn(96, 32, device=device),
        torch.softmax(torch.randn(96, device=device), dim=0),
        32,
        "what object is shown",
        EC_QMO_LAMBDA_H="1.0",
        EC_QMO_LAMBDA_P="0.0",
    )
    assert abs(unnormalized_info["D_q"] - unnormalized_info["entropy"]) < 1e-6


def test_qfid_selector_default_and_explicit_residual_match(ECPruner, device):
    visual_tokens = torch.randn(1, 96, 32, device=device)
    semantic_response = torch.softmax(torch.randn(96, 4, device=device), dim=-1)
    common = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_SELECT_MODE": "qf",
        "EC_QFID_PROB_SOURCE": "semantic",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_TAU": "0.50",
        "EC_QFID_DEPOLARIZE_MODE": "fixed",
        "EC_QFID_DEPOLARIZE": "0.15",
        "EC_QFID_BUDGET_CALIB": "0",
        "EC_QFID_SPECTRAL_FILTER": "0",
        "EC_QFID_SPATIAL_STATE": "0",
        "EC_QFID_MEASURE_PRIOR_MODE": "none",
        "EC_QFID_ANCHOR_MODE": "none",
        "EC_USE_SEMANTIC_CANDIDATE": "0",
        "EC_USE_SPATIAL_CANDIDATE": "0",
        "EC_USE_REPULSION": "0",
        "EC_USE_COMPLEMENT": "0",
        "EC_USE_PHI": "0",
        "EC_USE_CHAIN_COVERAGE": "0",
    }
    previous_selector = os.environ.pop("EC_QFID_SELECTOR", None)
    try:
        with env_override(**common):
            default_pruner = ECPruner(debug=False)
            default_idx, default_info = default_pruner.select(
                visual_tokens,
                keep_num=32,
                b=semantic_response,
            )
    finally:
        if previous_selector is not None:
            os.environ["EC_QFID_SELECTOR"] = previous_selector

    with env_override(**common, EC_QFID_SELECTOR="qfi_residual"):
        residual_pruner = ECPruner(debug=False)
        residual_idx, residual_info = residual_pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
        )

    assert torch.equal(default_idx, residual_idx)
    assert default_info["solver_used"] == "qfid_residual_greedy"
    assert residual_info["solver_used"] == "qfid_residual_greedy"
    assert default_info["qfid_info"]["selector"] == "qfi_residual"
    assert residual_info["qfid_info"]["selector"] == "qfi_residual"


def test_qfid_evidence_recovery_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="qf",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.105",
        EC_QFID_CLS_ATTN_LAYER="-2",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_CLS_GATE="1",
        EC_QFID_CLS_GATE_MODE="agreement",
        EC_QFID_CLS_GATE_BETA_BASE="0.105",
        EC_QFID_CLS_GATE_MIN="0.5",
        EC_QFID_CLS_GATE_MAX="1.5",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SELECTOR="evidence_recover",
        EC_QFID_BUDGET_CALIB="0",
        EC_QFID_SPECTRAL_FILTER="0",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        num_tokens = 96
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, num_tokens, device=device)
        for keep_num in (32, 64):
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
            )
            assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
            assert debug_info["selection_formulation"] == "uncovered_evidence_recovery"
            assert debug_info["solver_used"] == "evidence_recover_greedy"
            qfid_info = debug_info["qfid_info"]
            assert qfid_info["selector"] == "evidence_recover"
            assert qfid_info["budget_calib"] is False
            assert qfid_info["spectral_filter_enabled"] is False
            assert qfid_info["final_keep_size"] == keep_num
            assert 0.0 <= qfid_info["evidence_coverage_min"] <= 1.0
            assert 0.0 <= qfid_info["evidence_coverage_mean"] <= 1.0
            assert qfid_info["evidence_coverage_max"] <= 1.0 + 1e-6
            assert qfid_info["evidence_uncovered_mass"] >= 0.0
            assert_finite_debug_values(qfid_info)


def test_qfid_core_then_recover_select(ECPruner, device):
    common = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_SELECT_MODE": "qf",
        "EC_QFID_PROB_SOURCE": "clsmix",
        "EC_QFID_CLS_MIX_MODE": "linear",
        "EC_QFID_CLS_MIX_BETA": "0.105",
        "EC_QFID_CLS_ATTN_LAYER": "-2",
        "EC_QFID_CLS_HEAD_REDUCE": "mean",
        "EC_QFID_CLS_GATE": "1",
        "EC_QFID_CLS_GATE_MODE": "agreement",
        "EC_QFID_CLS_GATE_BETA_BASE": "0.105",
        "EC_QFID_CLS_GATE_MIN": "0.5",
        "EC_QFID_CLS_GATE_MAX": "1.5",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_TAU": "0.50",
        "EC_QFID_DEPOLARIZE_MODE": "fixed",
        "EC_QFID_DEPOLARIZE": "0.15",
        "EC_QFID_BUDGET_CALIB": "0",
        "EC_QFID_SPECTRAL_FILTER": "0",
        "EC_QFID_SPATIAL_STATE": "0",
        "EC_QFID_MEASURE_PRIOR_MODE": "none",
        "EC_QFID_ANCHOR_MODE": "none",
        "EC_USE_SEMANTIC_CANDIDATE": "0",
        "EC_USE_SPATIAL_CANDIDATE": "0",
        "EC_USE_REPULSION": "0",
        "EC_USE_COMPLEMENT": "0",
        "EC_USE_PHI": "0",
        "EC_USE_CHAIN_COVERAGE": "0",
    }
    num_tokens = 96
    visual_tokens = torch.randn(1, num_tokens, 32, device=device)
    semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
    cls_attention = torch.rand(1, num_tokens, device=device)

    with env_override(**common, EC_QFID_SELECTOR="qfi_residual"):
        residual_pruner = ECPruner(debug=False)
        residual_idx, residual_info = residual_pruner.select(
            visual_tokens,
            keep_num=64,
            b=semantic_response,
            cls_attn=cls_attention,
        )

    with env_override(**common, EC_QFID_SELECTOR="evidence_recover"):
        evidence_pruner = ECPruner(debug=False)
        evidence_idx, evidence_info = evidence_pruner.select(
            visual_tokens,
            keep_num=64,
            b=semantic_response,
            cls_attn=cls_attention,
        )

    for ratio, expected_core in (("0.5", 32), ("0.75", 48)):
        with env_override(**common, EC_QFID_SELECTOR="core_then_recover", EC_QFID_CORE_RATIO=ratio):
            pruner = ECPruner(debug=False)
            for keep_num in (32, 64):
                keep_idx, debug_info = pruner.select(
                    visual_tokens,
                    keep_num=keep_num,
                    b=semantic_response,
                    cls_attn=cls_attention,
                )
                assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
                assert debug_info["selection_formulation"] == "core_then_uncovered_evidence_recovery"
                assert debug_info["solver_used"] == "core_then_recover_greedy"
                qfid_info = debug_info["qfid_info"]
                assert qfid_info["selector"] == "core_then_recover"
                assert qfid_info["budget_calib"] is False
                assert qfid_info["spectral_filter_enabled"] is False
                assert qfid_info["final_keep_size"] == keep_num
                assert 0.0 <= qfid_info["core_ratio"] <= 1.0
                assert qfid_info["core_size"] == min(expected_core if keep_num == 64 else int(round(keep_num * float(ratio))), keep_num)
                assert qfid_info["recover_size"] == keep_num - qfid_info["core_size"]
                assert 0.0 <= qfid_info["evidence_coverage_min"] <= 1.0
                assert 0.0 <= qfid_info["evidence_coverage_mean"] <= 1.0
                assert qfid_info["evidence_coverage_max"] <= 1.0 + 1e-6
                assert qfid_info["evidence_uncovered_mass"] >= 0.0
                assert_finite_debug_values(qfid_info)

    with env_override(**common, EC_QFID_SELECTOR="core_then_recover", EC_QFID_CORE_RATIO="0"):
        ratio0_pruner = ECPruner(debug=False)
        ratio0_idx, _ = ratio0_pruner.select(
            visual_tokens,
            keep_num=64,
            b=semantic_response,
            cls_attn=cls_attention,
        )
    assert torch.equal(ratio0_idx, evidence_idx)
    assert evidence_info["qfid_info"]["selector"] == "evidence_recover"

    with env_override(**common, EC_QFID_SELECTOR="core_then_recover", EC_QFID_CORE_RATIO="1"):
        ratio1_pruner = ECPruner(debug=False)
        ratio1_idx, _ = ratio1_pruner.select(
            visual_tokens,
            keep_num=64,
            b=semantic_response,
            cls_attn=cls_attention,
        )
    assert torch.equal(ratio1_idx, residual_idx)
    assert residual_info["qfid_info"]["selector"] == "qfi_residual"


def test_qfid_adaptive_core_recover_select(ECPruner, device):
    common = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_SELECT_MODE": "qf",
        "EC_QFID_PROB_SOURCE": "clsmix",
        "EC_QFID_CLS_MIX_MODE": "linear",
        "EC_QFID_CLS_MIX_BETA": "0.105",
        "EC_QFID_CLS_ATTN_LAYER": "-2",
        "EC_QFID_CLS_HEAD_REDUCE": "mean",
        "EC_QFID_CLS_GATE": "1",
        "EC_QFID_CLS_GATE_MODE": "agreement",
        "EC_QFID_CLS_GATE_BETA_BASE": "0.105",
        "EC_QFID_CLS_GATE_MIN": "0.5",
        "EC_QFID_CLS_GATE_MAX": "1.5",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_TAU": "0.50",
        "EC_QFID_DEPOLARIZE_MODE": "fixed",
        "EC_QFID_DEPOLARIZE": "0.15",
        "EC_QFID_SELECTOR": "adaptive_core_recover",
        "EC_QFID_ADAPT_MODE": "entropy_question",
        "EC_QFID_ADAPT_RECOVER_MIN_RATIO": "0.00",
        "EC_QFID_ADAPT_RECOVER_MAX_RATIO": "0.25",
        "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO": "0.125",
        "EC_QFID_ADAPT_RECOVER_BINARY_RATIO": "0.0625",
        "EC_QFID_ADAPT_RECOVER_OPEN_RATIO": "0.25",
        "EC_QFID_ADAPT_RECOVER_REL_RATIO": "0.25",
        "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO": "0.25",
        "EC_QFID_ADAPT_RECOVER_ATTR_RATIO": "0.125",
        "EC_QFID_ADAPT_RECOVER_CAP": "16",
        "EC_QFID_ADAPT_ENTROPY_GATE": "1",
        "EC_QFID_ADAPT_ENTROPY_LOW": "0.70",
        "EC_QFID_ADAPT_ENTROPY_HIGH": "0.95",
        "EC_QFID_RECOVER_PRIOR_ANCHOR": "1",
        "EC_QFID_RECOVER_PRIOR_GAMMA": "0.5",
        "EC_QFID_RECOVER_CAND_POOL": "1",
        "EC_QFID_RECOVER_CAND_MULT": "3.0",
        "EC_QFID_BUDGET_CALIB": "0",
        "EC_QFID_SPECTRAL_FILTER": "0",
        "EC_QFID_SPATIAL_STATE": "0",
        "EC_QFID_MEASURE_PRIOR_MODE": "none",
        "EC_QFID_ANCHOR_MODE": "none",
        "EC_USE_SEMANTIC_CANDIDATE": "0",
        "EC_USE_SPATIAL_CANDIDATE": "0",
        "EC_USE_REPULSION": "0",
        "EC_USE_COMPLEMENT": "0",
        "EC_USE_PHI": "0",
        "EC_USE_CHAIN_COVERAGE": "0",
    }
    with env_override(**common):
        pruner = ECPruner(debug=False)
        num_tokens = 96
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, num_tokens, device=device)
        for keep_num in (32, 64):
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
                question="What is next to the table?",
            )
            assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
            assert debug_info["selection_formulation"] == "adaptive_core_then_uncovered_evidence_recovery"
            assert debug_info["solver_used"] == "adaptive_core_recover_greedy"
            qfid_info = debug_info["qfid_info"]
            assert qfid_info["selector"] == "adaptive_core_recover"
            assert qfid_info["adapt_mode"] == "entropy_question"
            assert qfid_info["question_type"] == "relation"
            assert qfid_info["question_text_available"] is True
            assert qfid_info["budget_calib"] is False
            assert qfid_info["spectral_filter_enabled"] is False
            assert qfid_info["recover_prior_anchor"] is True
            assert qfid_info["recover_cand_pool"] is True
            assert qfid_info["adapt_k_recover"] <= 16
            assert qfid_info["adapt_k_core"] + qfid_info["adapt_k_recover"] == keep_num
            assert_finite_debug_values(qfid_info)

    with env_override(**common):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 80, 32, device=device)
        semantic_response = torch.softmax(torch.randn(80, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(visual_tokens, keep_num=32, b=semantic_response)
        assert_valid_keep_idx(keep_idx, 32, 80, device)
        assert debug_info["qfid_info"]["adapt_mode"] == "entropy_only"
        assert debug_info["qfid_info"]["question_type"] == "default"
        assert debug_info["qfid_info"]["question_text_available"] is False


def test_qfid_qmo_cr_select(ECPruner, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "qmo_indices.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_SELECT_MODE="qf",
            EC_QFID_PROB_SOURCE="clsmix",
            EC_QFID_CLS_MIX_MODE="linear",
            EC_QFID_CLS_MIX_BETA="0.105",
            EC_QFID_CLS_ATTN_LAYER="-2",
            EC_QFID_CLS_HEAD_REDUCE="mean",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE_MODE="fixed",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_SELECTOR="qmo_cr",
            EC_QFID_BUDGET_CALIB="0",
            EC_QFID_SPECTRAL_FILTER="0",
            EC_QFID_SPATIAL_STATE="0",
            EC_QFID_MEASURE_PRIOR_MODE="none",
            EC_QFID_ANCHOR_MODE="none",
            EC_QMO_USE_PURITY="1",
            EC_QMO_W_COV="1.0",
            EC_QMO_W_H="0.5",
            EC_QMO_LAMBDA_J="0.2",
            EC_QMO_RECOVER_CAP="16",
            EC_QMO_DUMP_INDICES=str(dump_path),
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner = ECPruner(debug=False)
            num_tokens = 96
            keep_num = 32
            visual_tokens = torch.randn(1, num_tokens, 32, device=device)
            semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
            cls_attention = torch.rand(1, num_tokens, device=device)
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
                question="how many words are written on the sign",
            )

        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert debug_info["selection_formulation"] == "qmo_cr_coupled_recovery"
        assert debug_info["solver_used"] == "qmo_cr_greedy"
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["selector"] == "qmo_cr"
        assert qfid_info["qmo_cr_enabled"] is True
        assert qfid_info["qmo_use_purity"] is True
        assert qfid_info["adapt_k_core"] + qfid_info["adapt_k_recover"] == keep_num
        assert 0.0 <= qfid_info["qmo_entropy"] <= 1.0
        assert 0.0 <= qfid_info["qmo_purity"] <= 1.0
        assert 0.0 <= qfid_info["qmo_D_q"] <= 1.0
        assert_finite_debug_values(qfid_info)
        rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["K"] == keep_num
        assert len(rows[0]["selected_indices"]) == keep_num

    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="qf",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SELECTOR="qmo_cr",
        EC_QFID_BUDGET_CALIB="0",
        EC_QFID_SPECTRAL_FILTER="0",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(2, 48, 16, device=device)
        semantic_response = torch.softmax(torch.randn(48, 4, device=device), dim=-1)
        try:
            pruner.select(visual_tokens, keep_num=16, b=semantic_response, question="what color is the bus")
        except ValueError as exc:
            assert "requires batch size 1" in str(exc)
        else:
            raise AssertionError("qmo_cr should reject batch size > 1")


def test_qmo_gated_selector_function(device):
    infer_gate_type, selector = load_qmo_gated_helpers()
    assert infer_gate_type("Are there more cars than buses?") == "compare"
    assert infer_gate_type("What kind of animal is shown?") == "category"
    assert infer_gate_type("What object is next to the chair?") == "object"
    assert infer_gate_type("What color is the bus?") == "default"
    assert infer_gate_type(None) == "default"

    states = torch.randn(96, 32, device=device)
    qmo_states = torch.randn(96, 32, device=device)
    probs = torch.softmax(torch.randn(96, device=device), dim=0)
    qfi_order = torch.randperm(96, device=device)
    common = dict(
        EC_QFID_ADAPT_MODE="entropy_question",
        EC_QFID_ADAPT_RECOVER_MIN_RATIO="0.00",
        EC_QFID_ADAPT_RECOVER_MAX_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO="0.125",
        EC_QFID_ADAPT_RECOVER_BINARY_RATIO="0.0625",
        EC_QFID_ADAPT_RECOVER_OPEN_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_REL_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_COMPARE_RATIO="0.25",
        EC_QFID_ADAPT_RECOVER_ATTR_RATIO="0.125",
        EC_QFID_ADAPT_RECOVER_CAP="16",
        EC_QFID_ADAPT_ENTROPY_GATE="0",
        EC_QFID_RECOVER_PRIOR_ANCHOR="1",
        EC_QFID_RECOVER_PRIOR_GAMMA="0.5",
        EC_QFID_RECOVER_CAND_POOL="0",
        EC_QMO_GATE_TYPES="compare,category,object",
        EC_QMO_GATE_BETA="0.1",
        EC_QMO_W_H="0.1",
        EC_QMO_LAMBDA_J="0.01",
        EC_QMO_USE_PURITY="0",
    )

    with env_override(**common, EC_QMO_GATE_MODE="recovery_only"):
        default_idx, default_info = selector(
            states, probs, 32, qfi_order, question_text="What color is the bus?", qmo_states=qmo_states
        )
        compare_idx, compare_info = selector(
            states, probs, 32, qfi_order, question_text="Are there more cars than buses?", qmo_states=qmo_states
        )
    assert_valid_keep_idx(default_idx, 32, 96, device)
    assert_valid_keep_idx(compare_idx, 32, 96, device)
    assert default_info["gate_enabled"] is False
    assert default_info["baseline_overlap"] == 1.0
    assert compare_info["gate_enabled"] is True
    assert compare_info["gate_mode"] == "recovery_only"
    assert 0.0 <= compare_info["baseline_overlap"] <= 1.0
    assert 0.0 <= compare_info["qmo_cr_overlap"] <= 1.0

    with env_override(**common, EC_QMO_GATE_MODE="full"):
        full_idx, full_info = selector(
            states, probs, 32, qfi_order, question_text="What type of vehicle is present?", qmo_states=qmo_states
        )
    assert_valid_keep_idx(full_idx, 32, 96, device)
    assert full_info["gate_enabled"] is True
    assert full_info["gate_mode"] == "full"
    assert full_info["qmo_cr_overlap"] == 1.0


def test_qfid_qmo_gated_select(ECPruner, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "qmo_gated_indices.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_SELECT_MODE="qf",
            EC_QFID_PROB_SOURCE="clsmix",
            EC_QFID_CLS_MIX_MODE="linear",
            EC_QFID_CLS_MIX_BETA="0.105",
            EC_QFID_CLS_ATTN_LAYER="-2",
            EC_QFID_CLS_HEAD_REDUCE="mean",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE_MODE="fixed",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_SELECTOR="qmo_gated",
            EC_QFID_ADAPT_MODE="entropy_question",
            EC_QFID_ADAPT_RECOVER_MIN_RATIO="0.00",
            EC_QFID_ADAPT_RECOVER_MAX_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO="0.125",
            EC_QFID_ADAPT_RECOVER_COMPARE_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_CAP="16",
            EC_QFID_ADAPT_ENTROPY_GATE="0",
            EC_QFID_RECOVER_PRIOR_ANCHOR="1",
            EC_QFID_RECOVER_PRIOR_GAMMA="0.5",
            EC_QFID_RECOVER_CAND_POOL="0",
            EC_QFID_BUDGET_CALIB="0",
            EC_QFID_SPECTRAL_FILTER="0",
            EC_QFID_SPATIAL_STATE="0",
            EC_QFID_MEASURE_PRIOR_MODE="none",
            EC_QFID_ANCHOR_MODE="none",
            EC_QMO_GATE_MODE="recovery_only",
            EC_QMO_GATE_TYPES="compare,category,object",
            EC_QMO_GATE_BETA="0.1",
            EC_QMO_W_H="0.1",
            EC_QMO_LAMBDA_J="0.01",
            EC_QMO_USE_PURITY="0",
            EC_QMO_DUMP_INDICES=str(dump_path),
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner = ECPruner(debug=False)
            num_tokens = 96
            keep_num = 32
            visual_tokens = torch.randn(1, num_tokens, 32, device=device)
            semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
            cls_attention = torch.rand(1, num_tokens, device=device)
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
                question="Are there more cars than buses?",
            )

        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert debug_info["selection_formulation"] == "qmo_gated_adaptive_recovery"
        assert debug_info["solver_used"] == "qmo_gated_greedy"
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["selector"] == "qmo_gated"
        assert qfid_info["qmo_gate_enabled"] is True
        assert qfid_info["qmo_gate_type"] == "compare"
        assert qfid_info["qmo_gate_mode"] == "recovery_only"
        assert 0.0 <= qfid_info["qmo_gate_baseline_overlap"] <= 1.0
        assert 0.0 <= qfid_info["qmo_gate_qmo_cr_overlap"] <= 1.0
        rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["selector"] == "qmo_gated"
        assert rows[0]["gate_enabled"] is True
        assert rows[0]["gate_type"] == "compare"
        assert len(rows[0]["selected_indices"]) == keep_num
        assert len(rows[0]["baseline_indices"]) == keep_num
        assert len(rows[0]["qmo_cr_indices"]) == keep_num


def test_qsp_density_projective_selector_function(device):
    selector = load_select_by_qsp_density_projective()
    states = torch.randn(96, 32, device=device)
    probs = torch.softmax(torch.randn(96, device=device), dim=0)
    qfi_order = torch.randperm(96, device=device)

    variants = [
        ("density_projection", "complement_projected"),
        ("top_p", "prior_recovery"),
        ("pairwise_fidelity", "none"),
    ]
    for core_selection, recovery in variants:
        with env_override(
            EC_QSP_CORE_SELECTION=core_selection,
            EC_QSP_RECOVERY=recovery,
            EC_QSP_USE_SPATIAL_LOCAL_PRIOR="0",
            EC_QSP_LAMBDA_B="0.1",
            EC_QSP_CANDIDATE_TOPM="48",
            EC_QSP_CORE_RATIO="0.75",
            EC_QSP_RECOVER_CAP="16",
        ):
            keep_idx, info = selector(states, probs, 32, qfi_core_order=qfi_order)
        assert_valid_keep_idx(keep_idx, 32, 96, device)
        assert info["qsp_enabled"] is True
        assert info["qsp_core_selection"] == core_selection
        assert info["qsp_recovery"] == recovery
        assert info["qsp_k_core"] + info["qsp_k_recover"] <= 32
        assert len(info["selected_indices"]) == 32
        assert info["qsp_candidate_size"] == 48


def test_qfid_qsp_density_projective_select(ECPruner, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "qsp_indices.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_SELECT_MODE="qf",
            EC_QFID_PROB_SOURCE="clsmix",
            EC_QFID_CLS_MIX_MODE="linear",
            EC_QFID_CLS_MIX_BETA="0.105",
            EC_QFID_CLS_ATTN_LAYER="-2",
            EC_QFID_CLS_HEAD_REDUCE="mean",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE_MODE="fixed",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_SELECTOR="qsp_cr_density_projective",
            EC_QFID_BUDGET_CALIB="0",
            EC_QFID_SPECTRAL_FILTER="0",
            EC_QFID_SPATIAL_STATE="0",
            EC_QFID_MEASURE_PRIOR_MODE="none",
            EC_QFID_ANCHOR_MODE="none",
            EC_QSP_CORE_SELECTION="density_projection",
            EC_QSP_RECOVERY="complement_projected",
            EC_QSP_USE_SPATIAL_LOCAL_PRIOR="0",
            EC_QSP_LAMBDA_B="0.1",
            EC_QSP_CANDIDATE_TOPM="0",
            EC_QSP_CORE_RATIO="0.75",
            EC_QSP_RECOVER_CAP="16",
            EC_QSP_DUMP_INDICES=str(dump_path),
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner = ECPruner(debug=False)
            num_tokens = 96
            keep_num = 32
            visual_tokens = torch.randn(1, num_tokens, 32, device=device)
            semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
            cls_attention = torch.rand(1, num_tokens, device=device)
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
                question="What object is next to the bus?",
            )

        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert debug_info["selection_formulation"] == "density_projective_core_recover"
        assert debug_info["solver_used"] == "qsp_density_projective_greedy"
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["selector"] == "qsp_cr_density_projective"
        assert qfid_info["qsp_enabled"] is True
        assert qfid_info["qsp_fallback"] is False
        assert qfid_info["qsp_core_selection"] == "density_projection"
        assert qfid_info["qsp_recovery"] == "complement_projected"
        assert qfid_info["qsp_k_core"] + qfid_info["qsp_k_recover"] <= keep_num
        assert_finite_debug_values(qfid_info)
        rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["selector"] == "qsp_cr_density_projective"
        assert rows[0]["core_selection"] == "density_projection"
        assert rows[0]["recovery"] == "complement_projected"
        assert len(rows[0]["selected_indices"]) == keep_num


def test_qfid_qfi_core_complement_recovery_select(ECPruner, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "qfi_core_complement_indices.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_SELECT_MODE="qf",
            EC_QFID_PROB_SOURCE="clsmix",
            EC_QFID_CLS_MIX_MODE="linear",
            EC_QFID_CLS_MIX_BETA="0.105",
            EC_QFID_CLS_ATTN_LAYER="-2",
            EC_QFID_CLS_HEAD_REDUCE="mean",
            EC_QFID_CLS_GATE="1",
            EC_QFID_CLS_GATE_MODE="agreement",
            EC_QFID_CLS_GATE_BETA_BASE="0.105",
            EC_QFID_CLS_GATE_MIN="0.5",
            EC_QFID_CLS_GATE_MAX="1.5",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE_MODE="fixed",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_SELECTOR="qfi_core_complement_recovery",
            EC_QFID_ADAPT_MODE="entropy_question",
            EC_QFID_ADAPT_RECOVER_MIN_RATIO="0.00",
            EC_QFID_ADAPT_RECOVER_MAX_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO="0.125",
            EC_QFID_ADAPT_RECOVER_BINARY_RATIO="0.0625",
            EC_QFID_ADAPT_RECOVER_OPEN_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_REL_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_COMPARE_RATIO="0.25",
            EC_QFID_ADAPT_RECOVER_ATTR_RATIO="0.125",
            EC_QFID_ADAPT_RECOVER_CAP="16",
            EC_QFID_ADAPT_ENTROPY_GATE="1",
            EC_QFID_ADAPT_ENTROPY_LOW="0.70",
            EC_QFID_ADAPT_ENTROPY_HIGH="0.95",
            EC_QFID_RECOVER_PRIOR_ANCHOR="1",
            EC_QFID_RECOVER_PRIOR_GAMMA="0.5",
            EC_QFID_RECOVER_CAND_POOL="0",
            EC_QFID_BUDGET_CALIB="0",
            EC_QFID_SPECTRAL_FILTER="0",
            EC_QFID_SPATIAL_STATE="0",
            EC_QFID_MEASURE_PRIOR_MODE="none",
            EC_QFID_ANCHOR_MODE="none",
            EC_QSP_DUMP_INDICES=str(dump_path),
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner = ECPruner(debug=False)
            num_tokens = 96
            keep_num = 32
            visual_tokens = torch.randn(1, num_tokens, 32, device=device)
            semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
            cls_attention = torch.rand(1, num_tokens, device=device)
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
                question="What color is the bus?",
            )

        assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
        assert debug_info["selection_formulation"] == "qfi_core_complement_recovery"
        assert debug_info["solver_used"] == "qfi_core_complement_recovery_greedy"
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["selector"] == "qfi_core_complement_recovery"
        assert qfid_info["qsp_enabled"] is True
        assert qfid_info["qsp_core_selection"] == "qfi_core"
        assert qfid_info["qsp_recovery"] == "complement_projected"
        rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["selector"] == "qfi_core_complement_recovery"


def test_qfid_visual_kcenter_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="visual_kcenter",
        EC_QFID_PROB_SOURCE="uniform",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        num_tokens = 80
        keep_num = 16
        visual_tokens = torch.randn(1, num_tokens, 32, device=device)
        first_pruner = ECPruner(debug=False)
        second_pruner = ECPruner(debug=False)
        first_idx, first_info = first_pruner.select(visual_tokens, keep_num=keep_num)
        second_idx, second_info = second_pruner.select(visual_tokens, keep_num=keep_num)
        assert_valid_keep_idx(first_idx, keep_num, num_tokens, device)
        assert torch.equal(first_idx, second_idx)
        assert first_info["selection_formulation"] == "visual_cosine_kcenter"
        assert first_info["solver_used"] == "visual_kcenter"
        assert first_info["qfid_info"]["first_pivot"] == second_info["qfid_info"]["first_pivot"]


def test_qfid_gate_sidecar(ECPruner, device):
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_path = Path(temp_dir) / "gate_stats.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_SELECT_MODE="qf",
            EC_QFID_PROB_SOURCE="clsmix",
            EC_QFID_CLS_MIX_MODE="linear",
            EC_QFID_CLS_MIX_BETA="0.105",
            EC_QFID_CLS_GATE="1",
            EC_QFID_CLS_GATE_MODE="agreement",
            EC_QFID_CLS_GATE_BETA_BASE="0.105",
            EC_QFID_CLS_GATE_MIN="0.5",
            EC_QFID_CLS_GATE_MAX="1.5",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE_MODE="fixed",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_DEBUG_STATS_JSONL=str(stats_path),
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner = ECPruner(debug=False)
            num_tokens = 96
            keep_num = 24
            visual_tokens = torch.randn(1, num_tokens, 32, device=device)
            semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
            cls_attention = torch.rand(1, num_tokens, device=device)
            keep_idx, _ = pruner.select(
                visual_tokens,
                keep_num=keep_num,
                b=semantic_response,
                cls_attn=cls_attention,
            )
            assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)

        records = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        record = records[0]
        required = {
            "sample_index", "agreement", "js_div", "gate", "beta_eff", "beta_base",
            "p_sem_entropy", "p_cls_entropy", "keep_size", "budget_calib",
            "budget_alpha", "budget_target_neff", "budget_neff_before",
            "budget_neff_after", "budget_gamma",
        }
        assert required.issubset(record)
        assert 0.0 <= record["agreement"] <= 1.0
        assert 0.0525 - 1e-8 <= record["beta_eff"] <= 0.1575 + 1e-8
        assert record["keep_size"] == keep_num
        assert record["budget_calib"] is False


def test_qfid_budget_calibration_function(device):
    calibrate = load_budget_calibrate_prob()
    base_logits = torch.linspace(-4.0, 4.0, steps=576, device=device)
    base_prob = torch.softmax(base_logits, dim=-1)

    for keep_num in (32, 64, 128):
        for alpha in (2.0, 3.0, 4.0):
            calibrated, stats = calibrate(base_prob, keep_num=keep_num, alpha=alpha)
            assert calibrated.shape == base_prob.shape
            assert calibrated.device.type == device.type
            assert calibrated.dtype == base_prob.dtype
            assert torch.isfinite(calibrated).all()
            assert torch.allclose(calibrated.sum(), torch.tensor(1.0, device=device), atol=1e-6)
            before_error = abs(stats["budget_neff_before"] - stats["budget_target_neff"])
            after_error = abs(stats["budget_neff_after"] - stats["budget_target_neff"])
            assert after_error <= before_error + 1e-4
            assert 0.05 - 1e-8 <= stats["budget_gamma"] <= 20.0 + 1e-8

    batch_prob = torch.stack((base_prob, torch.flip(base_prob, dims=(0,))), dim=0)
    calibrated, stats = calibrate(batch_prob, keep_num=64, alpha=3.0)
    assert calibrated.shape == batch_prob.shape
    assert calibrated.dtype == batch_prob.dtype
    assert torch.isfinite(calibrated).all()
    assert torch.allclose(calibrated.sum(dim=-1), torch.ones(2, device=device), atol=1e-6)
    assert len(stats["budget_neff_before"]) == 2
    assert len(stats["budget_neff_after"]) == 2
    assert len(stats["budget_gamma"]) == 2


def test_qfid_budget_calibration_disabled_matches_legacy(ECPruner, device):
    visual_tokens = torch.randn(1, 96, 32, device=device)
    semantic_response = torch.softmax(torch.randn(96, 4, device=device), dim=-1)

    common = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_PROB_SOURCE": "semantic",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_TAU": "0.50",
        "EC_QFID_DEPOLARIZE": "0.15",
        "EC_QFID_BUDGET_CALIB": "0",
    }
    with env_override(**common, EC_QFID_BUDGET_ALPHA="2.0"):
        first_pruner = ECPruner(debug=False)
        first_idx, first_info = first_pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
        )
    with env_override(**common, EC_QFID_BUDGET_ALPHA="4.0"):
        second_pruner = ECPruner(debug=False)
        second_idx, second_info = second_pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
        )

    assert torch.equal(first_idx, second_idx)
    assert first_info["qfid_info"]["budget_calib"] is False
    assert second_info["qfid_info"]["budget_calib"] is False


def test_qfid_budget_calibration_select_and_sidecar(ECPruner, device):
    with tempfile.TemporaryDirectory() as temp_dir:
        stats_path = Path(temp_dir) / "budget_stats.jsonl"
        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_PROB_SOURCE="semantic",
            EC_QFID_KERNEL="density",
            EC_QFID_TAU="0.50",
            EC_QFID_DEPOLARIZE="0.15",
            EC_QFID_BUDGET_CALIB="1",
            EC_QFID_BUDGET_ALPHA="3.0",
            EC_QFID_BUDGET_GAMMA_MIN="0.05",
            EC_QFID_BUDGET_GAMMA_MAX="20.0",
            EC_QFID_BUDGET_ITERS="30",
            EC_QFID_BUDGET_EPS="1e-12",
            EC_QFID_DEBUG_STATS_JSONL=str(stats_path),
        ):
            pruner = ECPruner(debug=False)
            visual_tokens = torch.randn(1, 96, 32, device=device)
            semantic_response = torch.softmax(torch.randn(96, 4, device=device), dim=-1)
            keep_idx, debug_info = pruner.select(
                visual_tokens,
                keep_num=32,
                b=semantic_response,
            )

        assert_valid_keep_idx(keep_idx, 32, 96, device)
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["budget_calib"] is True
        assert qfid_info["budget_target_neff"] == 96.0
        assert torch.isfinite(torch.tensor(qfid_info["budget_neff_after"]))
        records = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert records[0]["budget_calib"] is True
        assert records[0]["budget_target_neff"] == 96.0


def test_spectral_filter_kernel_identity_and_trace(device):
    spectral_filter = load_spectral_filter_kernel()
    kernel = make_psd_kernel(24, device=device)
    filtered, info = spectral_filter(kernel, gamma=1.0, eps=1e-12, trace_norm=True)
    assert filtered.shape == kernel.shape
    assert filtered.device.type == kernel.device.type
    assert filtered.dtype == kernel.dtype
    assert torch.isfinite(filtered).all()
    assert torch.allclose(filtered, filtered.t(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(filtered, kernel, atol=1e-4, rtol=1e-4)
    assert abs(float(torch.trace(filtered).item()) - float(torch.trace(kernel).item())) <= 1e-4
    assert info["spectral_filter"] is True
    assert info["spectral_trace_norm"] is True
    assert info["spectral_rank_pos"] > 0


def test_spectral_filter_kernel_gamma_variants(device):
    spectral_filter = load_spectral_filter_kernel()
    kernel = make_psd_kernel(20, device=device, dtype=torch.float32)
    trace_ref = float(torch.trace(kernel).item())
    for gamma in (0.5, 0.75, 1.25):
        filtered, info = spectral_filter(kernel, gamma=gamma, eps=1e-12, trace_norm=True)
        assert filtered.shape == kernel.shape
        assert filtered.dtype == kernel.dtype
        assert filtered.device.type == kernel.device.type
        assert torch.isfinite(filtered).all()
        assert torch.allclose(filtered, filtered.t(), atol=1e-6, rtol=1e-6)
        assert float(torch.diag(filtered).min().item()) >= -1e-6
        assert abs(float(torch.trace(filtered).item()) - trace_ref) <= 1e-4
        assert info["spectral_gamma"] == gamma


def test_spectral_filter_rank_deficient_kernel(device):
    spectral_filter = load_spectral_filter_kernel()
    kernel = make_psd_kernel(16, device=device, rank=3)
    filtered, info = spectral_filter(kernel, gamma=0.75, eps=1e-12, trace_norm=True)
    assert filtered.shape == kernel.shape
    assert torch.isfinite(filtered).all()
    assert torch.allclose(filtered, filtered.t(), atol=1e-6, rtol=1e-6)
    assert info["spectral_rank_pos"] <= 3 + 1


def test_spectral_filter_preserves_dtype_and_device(device):
    spectral_filter = load_spectral_filter_kernel()
    for dtype in (torch.float32, torch.float64):
        kernel = make_psd_kernel(12, device=device, dtype=dtype)
        filtered, _ = spectral_filter(kernel, gamma=0.5, eps=1e-12, trace_norm=False)
        assert filtered.dtype == kernel.dtype
        assert filtered.device.type == kernel.device.type


def test_qfid_spectral_disabled_matches_legacy(ECPruner, device):
    visual_tokens = torch.randn(1, 96, 32, device=device)
    semantic_response = torch.softmax(torch.randn(96, 4, device=device), dim=-1)
    cls_attention = torch.rand(1, 96, device=device)
    common = {
        "EC_SCORE_SOURCE": "qfid",
        "EC_QFID_SELECT_MODE": "qf",
        "EC_QFID_PROB_SOURCE": "clsmix",
        "EC_QFID_CLS_MIX_MODE": "linear",
        "EC_QFID_CLS_MIX_BETA": "0.105",
        "EC_QFID_CLS_ATTN_LAYER": "-2",
        "EC_QFID_CLS_HEAD_REDUCE": "mean",
        "EC_QFID_CLS_GATE": "1",
        "EC_QFID_CLS_GATE_MODE": "agreement",
        "EC_QFID_CLS_GATE_BETA_BASE": "0.105",
        "EC_QFID_CLS_GATE_MIN": "0.5",
        "EC_QFID_CLS_GATE_MAX": "1.5",
        "EC_QFID_KERNEL": "density",
        "EC_QFID_TAU": "0.50",
        "EC_QFID_DEPOLARIZE_MODE": "fixed",
        "EC_QFID_DEPOLARIZE": "0.15",
        "EC_QFID_BUDGET_CALIB": "0",
        "EC_QFID_SPECTRAL_FILTER": "0",
    }
    with env_override(**common, EC_QFID_SPECTRAL_GAMMA="0.5"):
        first_pruner = ECPruner(debug=False)
        first_idx, first_info = first_pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
            cls_attn=cls_attention,
        )
    with env_override(**common, EC_QFID_SPECTRAL_GAMMA="1.25"):
        second_pruner = ECPruner(debug=False)
        second_idx, second_info = second_pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
            cls_attn=cls_attention,
        )
    assert torch.equal(first_idx, second_idx)
    assert first_info["qfid_info"]["spectral_filter_enabled"] is False
    assert second_info["qfid_info"]["spectral_filter_enabled"] is False


def test_qfid_spectral_density_select(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_SELECT_MODE="qf",
        EC_QFID_PROB_SOURCE="clsmix",
        EC_QFID_CLS_MIX_MODE="linear",
        EC_QFID_CLS_MIX_BETA="0.105",
        EC_QFID_CLS_ATTN_LAYER="-2",
        EC_QFID_CLS_HEAD_REDUCE="mean",
        EC_QFID_CLS_GATE="1",
        EC_QFID_CLS_GATE_MODE="agreement",
        EC_QFID_CLS_GATE_BETA_BASE="0.105",
        EC_QFID_CLS_GATE_MIN="0.5",
        EC_QFID_CLS_GATE_MAX="1.5",
        EC_QFID_KERNEL="density",
        EC_QFID_TAU="0.50",
        EC_QFID_DEPOLARIZE_MODE="fixed",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_BUDGET_CALIB="0",
        EC_QFID_SPECTRAL_FILTER="1",
        EC_QFID_SPECTRAL_GAMMA="0.75",
        EC_QFID_SPECTRAL_EPS="1e-12",
        EC_QFID_SPECTRAL_TRACE_NORM="1",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 96, 32, device=device)
        semantic_response = torch.softmax(torch.randn(96, 4, device=device), dim=-1)
        cls_attention = torch.rand(1, 96, device=device)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=32,
            b=semantic_response,
            cls_attn=cls_attention,
        )
        assert_valid_keep_idx(keep_idx, 32, 96, device)
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["spectral_filter_enabled"] is True
        assert qfid_info["spectral_filter_applied"] is True
        assert qfid_info["spectral_filter_fallback"] is False
        assert abs(qfid_info["spectral_gamma"] - 0.75) <= 1e-8
        assert qfid_info["final_keep_size"] == 32
        assert_finite_debug_values(qfid_info)


def test_qfid_spectral_amplitude_fallback(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_PROB_SOURCE="uniform",
        EC_QFID_KERNEL="amplitude",
        EC_QFID_SPECTRAL_FILTER="1",
        EC_QFID_SPECTRAL_GAMMA="0.75",
        EC_QFID_DEPOLARIZE="0.0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 64, 16, device=device)
        keep_idx, debug_info = pruner.select(visual_tokens, keep_num=16)
        assert_valid_keep_idx(keep_idx, 16, 64, device)
        qfid_info = debug_info["qfid_info"]
        assert qfid_info["spectral_filter_enabled"] is True
        assert qfid_info["spectral_filter_applied"] is False
        assert qfid_info["spectral_filter_fallback"] is True


def test_qfid_cls_entropy_weighted_reduce(ECPruner, device):
    cls_attention = torch.tensor(
        [[[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]]],
        dtype=torch.float32,
        device=device,
    )
    with env_override(EC_QFID_CLS_HEAD_REDUCE="mean"):
        mean_pruner = ECPruner(debug=False)
        p_mean, _ = mean_pruner.compute_qfid_cls_probability(cls_attention, 4, device)
    with env_override(EC_QFID_CLS_HEAD_REDUCE="entropy_weighted"):
        weighted_pruner = ECPruner(debug=False)
        p_weighted, info = weighted_pruner.compute_qfid_cls_probability(cls_attention, 4, device)

    assert torch.isfinite(p_weighted).all()
    assert abs(float(p_weighted.sum().item()) - 1.0) <= 1e-6
    assert p_weighted[0] > p_mean[0]
    assert info["cls_head_reduce"] == "entropy_weighted"
    assert info["cls_head_weight_max"] > info["cls_head_weight_min"]


def test_clip_cls_attention_layer_reconstruction(device):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    CLIPVisionTower = load_clip_vision_tower_class()
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=8,
        patch_size=2,
        attention_dropout=0.0,
    )
    model = CLIPVisionModel(config).to(device).eval()
    tower = CLIPVisionTower.__new__(CLIPVisionTower)
    torch.nn.Module.__init__(tower)
    tower.vision_tower = model
    pixels = torch.randn(1, 3, 8, 8, device=device)
    with torch.no_grad():
        reference = model(pixels, output_hidden_states=True, output_attentions=True)
        expected = {
            "last": reference.attentions[-1][:, :, 0, 1:],
            "-2": reference.attentions[-2][:, :, 0, 1:],
            "-4": reference.attentions[-4][:, :, 0, 1:],
            "last4mean": torch.stack(
                [attention[:, :, 0, 1:] for attention in reference.attentions[-4:]],
                dim=0,
            ).mean(dim=0),
        }
        for layer_spec, expected_attention in expected.items():
            _, reconstructed = tower._forward_with_cls_attention(pixels, layer_spec=layer_spec)
            assert reconstructed is not None
            assert reconstructed.shape == expected_attention.shape
            assert torch.allclose(reconstructed, expected_attention.float(), atol=1e-5, rtol=1e-4)


def test_run_gqa_profile_contains_spectral_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    for profile_name in (
        "qfid_density_cg_eaqf_spectral_g05",
        "qfid_density_cg_eaqf_spectral_g075",
        "qfid_density_cg_eaqf_spectral_g125",
    ):
        assert profile_name in profile_script


def test_run_gqa_profile_contains_evidence_recover_profile():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    assert "qfi_evidence_recover_final" in profile_script
    assert "EC_QFID_SELECTOR=evidence_recover" in profile_script
    assert "EC_QFID_SELECTOR=qfi_residual" in profile_script
    assert 'PARAM="qfi_er_k${TOKEN}"' in gqa_script


def test_run_gqa_profile_contains_core_then_recover_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    assert "qfi_core_recover_r050" in profile_script
    assert "qfi_core_recover_r075" in profile_script
    assert "EC_QFID_SELECTOR=core_then_recover" in profile_script
    assert "EC_QFID_CORE_RATIO=${CORE_RATIO}" in profile_script
    assert "EC_QFID_BUDGET_CALIB=0" in profile_script
    assert "EC_QFID_SPECTRAL_FILTER=0" in profile_script
    assert 'PARAM="qfi_cr50_k${TOKEN}"' in gqa_script
    assert 'PARAM="qfi_cr75_k${TOKEN}"' in gqa_script


def test_run_gqa_profile_contains_adaptive_recover_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    for profile_name in (
        "qfi_adaptive_recover_budget",
        "qfi_adaptive_recover_prior",
        "qfi_adaptive_recover_final",
    ):
        assert profile_name in profile_script
    assert "EC_QFID_SELECTOR=adaptive_core_recover" in profile_script
    assert "EC_QFID_ADAPT_MODE=entropy_question" in profile_script
    assert "EC_QFID_ADAPT_RECOVER_CAP=16" in profile_script
    assert "EC_QFID_BUDGET_CALIB=0" in profile_script
    assert "EC_QFID_SPECTRAL_FILTER=0" in profile_script
    assert 'PARAM="qfi_adapt_bgt_k${TOKEN}"' in gqa_script
    assert 'PARAM="qfi_adapt_prior_k${TOKEN}"' in gqa_script
    assert 'PARAM="qfi_adapt_final_k${TOKEN}"' in gqa_script


def test_run_gqa_profile_contains_qmo_gated_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    assert "qmo_gated_adaptive_recover_prior" in profile_script
    assert "qmo_gated_adaptive_recover_prior_full" in profile_script
    assert "EC_QFID_SELECTOR=qmo_gated" in profile_script
    assert "EC_QMO_GATE_MODE=${QMO_GATE_MODE}" in profile_script
    assert 'QMO_GATE_TYPES="compare,category,object"' in profile_script
    assert "EC_QMO_GATE_TYPES=${QMO_GATE_TYPES}" in profile_script
    assert 'QMO_GATE_BETA="0.1"' in profile_script
    assert "EC_QMO_GATE_BETA=${QMO_GATE_BETA}" in profile_script
    assert 'PARAM="${EC_QMO_PROFILE}_k${TOKEN}"' in gqa_script


def test_run_gqa_profile_contains_qficr_ablation_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    for profile_name in (
        "qficr_ablate_full",
        "qficr_ablate_only_reduction",
        "qficr_ablate_random_restoration",
        "qficr_ablate_fixed_ratio",
        "qficr_ablate_no_spatial_local_prior",
        "qficr_ablate_no_spatial",
        "qficr_ablate_no_local",
        "qficr_ablate_sem_only",
        "qficr_ablate_cls_only",
        "qficr_ablate_fixed_fusion",
    ):
        assert profile_name in profile_script
    assert "EC_QFICR_RESTORATION_MODE=${QFICR_RESTORATION_MODE}" in profile_script
    assert "EC_QFICR_OBSERVATION_MODE=${QFICR_OBSERVATION_MODE}" in profile_script
    assert "EC_QFICR_ABLATION_PROFILE=${QFICR_ABLATION}" in profile_script
    assert 'PARAM="${PARAM}_qficr${EC_QFICR_ABLATION_TAG}"' in gqa_script


def test_run_gqa_profile_contains_qsp_density_projective_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    assert "qsp_cr_density_projective" in profile_script
    assert "density_projective_core_recover" in profile_script
    assert "qfi_core_complement_recovery" in profile_script
    assert "qsp_density_projective_task_anchor" in profile_script
    assert "qsp_density_projective_task_anchor_a05" in profile_script
    assert "qsp_density_projective_complement_bonus" in profile_script
    assert "EC_QFID_SELECTOR=${QSP_SELECTOR}" in profile_script
    assert "EC_QSP_CORE_SELECTION=${QSP_CORE_SELECTION}" in profile_script
    assert "EC_QSP_RECOVERY=${QSP_RECOVERY}" in profile_script
    assert "EC_QSP_TASK_ANCHOR_ALPHA=${QSP_TASK_ANCHOR_ALPHA}" in profile_script
    assert "EC_QSP_COMPLEMENT_BONUS_ETA=${QSP_COMPLEMENT_BONUS_ETA}" in profile_script
    assert "EC_QSP_DUMP_INDICES=logs/${QSP_PROFILE}_k${TOKEN}_indices.jsonl" in profile_script
    assert 'PARAM="${EC_QSP_PROFILE}_k${TOKEN}"' in gqa_script


def test_qfi_cards_selector_functions(device):
    zeno, spatial_buffer, progressive, coarse_to_fine = load_qfi_cards_helpers()
    states = torch.randn(96, 32, device=device)
    probs = torch.softmax(torch.randn(96, device=device), dim=0)
    qfi_order = torch.randperm(96, device=device)

    with env_override(EC_QFI_ZENO_BETA="0.10"):
        keep_idx, info, record = zeno(states, probs, 32, qfi_order, question_text="what is on the table")
    assert_valid_keep_idx(keep_idx, 32, 96, device)
    assert info["k_core"] + info["k_recover"] <= 32
    assert record["profile"] == "qfi_zeno_stabilized"
    assert record["zeno_beta"] == 0.10

    with env_override(EC_QFI_BUFFER_RATIO="0.15", EC_QFI_BUFFER_MAX="8", EC_QFI_BUFFER_MODE="8neighbor"):
        keep_idx, info, record = spatial_buffer(states, probs, 32, qfi_order, question_text="what color is it")
    assert_valid_keep_idx(keep_idx, 32, 96, device)
    assert record["profile"] == "qfi_spatial_buffer_recovery"
    assert record["buffer_mode"] == "8neighbor"

    with env_override(EC_QFI_PROGRESSIVE_MID_K="48", EC_QFI_PROGRESSIVE_FINAL_K="32"):
        keep_idx, info, record = progressive(states, probs, 32, qfi_order, question_text="is it red")
    assert_valid_keep_idx(keep_idx, 32, 96, device)
    assert record["progressive_mode"] == "candidate_fallback"
    assert record["mid_k"] == 48

    with env_override(EC_QFI_COARSE_K="48", EC_QFI_FINE_K="32"):
        keep_idx, info, record = coarse_to_fine(states, probs, 32, qfi_order, question_text="what type is it")
    assert_valid_keep_idx(keep_idx, 32, 96, device)
    assert record["profile"] == "qfi_coarse_to_fine_pruning"
    assert record["coarse_k"] == 48


def test_qfid_qfi_cards_select(ECPruner, device):
    variants = [
        "qfi_zeno_stabilized",
        "qfi_spatial_buffer_recovery",
        "qfi_progressive_128_to_64",
        "qfi_coarse_to_fine_pruning",
    ]
    for selector in variants:
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = Path(tmpdir) / f"{selector}.jsonl"
            with env_override(
                EC_SCORE_SOURCE="qfid",
                EC_QFID_SELECT_MODE="qf",
                EC_QFID_PROB_SOURCE="clsmix",
                EC_QFID_CLS_MIX_MODE="linear",
                EC_QFID_CLS_MIX_BETA="0.105",
                EC_QFID_CLS_ATTN_LAYER="-2",
                EC_QFID_CLS_HEAD_REDUCE="mean",
                EC_QFID_KERNEL="density",
                EC_QFID_TAU="0.50",
                EC_QFID_DEPOLARIZE_MODE="fixed",
                EC_QFID_DEPOLARIZE="0.15",
                EC_QFID_SELECTOR=selector,
                EC_QFID_ADAPT_MODE="entropy_question",
                EC_QFID_RECOVER_PRIOR_ANCHOR="1",
                EC_QFID_RECOVER_CAND_POOL="0",
                EC_QFI_PROGRESSIVE_MID_K="48",
                EC_QFI_PROGRESSIVE_FINAL_K="32",
                EC_QFI_COARSE_K="48",
                EC_QFI_FINE_K="32",
                EC_QFI_CARDS_DUMP_DIAG="1",
                EC_QFI_CARDS_DIAG_PATH=str(diag_path),
                EC_USE_SEMANTIC_CANDIDATE="0",
                EC_USE_SPATIAL_CANDIDATE="0",
                EC_USE_REPULSION="0",
                EC_USE_COMPLEMENT="0",
                EC_USE_PHI="0",
                EC_USE_CHAIN_COVERAGE="0",
            ):
                pruner = ECPruner(debug=False)
                num_tokens = 96
                keep_num = 32
                visual_tokens = torch.randn(1, num_tokens, 32, device=device)
                semantic_response = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1)
                cls_attention = torch.rand(1, num_tokens, device=device)
                keep_idx, debug_info = pruner.select(
                    visual_tokens,
                    keep_num=keep_num,
                    b=semantic_response,
                    cls_attn=cls_attention,
                    question="What object is next to the bus?",
                )
            assert_valid_keep_idx(keep_idx, keep_num, num_tokens, device)
            assert debug_info["solver_used"] == "qfi_cards_greedy"
            qfid_info = debug_info["qfid_info"]
            assert qfid_info["selector"] == selector
            assert qfid_info["qfi_cards_profile"] == selector
            assert qfid_info["qfi_cards_fallback"] is False
            rows = [json.loads(line) for line in diag_path.read_text().splitlines()]
            assert len(rows) == 1
            assert rows[0]["profile"] == selector
            assert len(rows[0]["final_indices"]) == keep_num


def test_run_gqa_profile_contains_qfi_cards_profiles():
    profile_script = (REPO_ROOT / "run_gqa_profile.sh").read_text()
    gqa_script = (REPO_ROOT / "scripts" / "v1_5" / "eval" / "gqa.sh").read_text()
    assert "qfi_zeno_stabilized" in profile_script
    assert "qfi_zeno_stabilized_b005" in profile_script
    assert "qfi_spatial_buffer_recovery" in profile_script
    assert "qfi_progressive_128_to_64" in profile_script
    assert "qfi_coarse_to_fine_pruning" in profile_script
    assert "EC_QFID_SELECTOR=${QFI_CARD_SELECTOR}" in profile_script
    assert "EC_QFI_CARDS_DIAG_PATH=logs/${QFI_CARD_PROFILE}_k${TOKEN}_cards_diag.jsonl" in profile_script
    assert 'PARAM="${EC_QFI_CARD_PROFILE}_k${TOKEN}"' in gqa_script


def test_qfid_depolarize_probability(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.25",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        b = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=b,
            semantic_response_source="clip",
        )
        assert source_used == "semantic"
        assert probs.shape == (576,)
        assert torch.isfinite(probs).all()
        assert float(probs.min().item()) >= 0.0
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert prob_info["mode"] == "fixed"
        assert abs(prob_info["adaptive_depolarize"] - 0.25) <= 1e-8

        with env_override(
            EC_SCORE_SOURCE="qfid",
            EC_QFID_TAU="0.50",
            EC_QFID_EPS="1e-6",
            EC_QFID_PROB_SOURCE="semantic",
            EC_QFID_KERNEL="density",
            EC_QFID_DEPOLARIZE="0.0",
            EC_USE_SEMANTIC_CANDIDATE="0",
            EC_USE_SPATIAL_CANDIDATE="0",
            EC_USE_REPULSION="0",
            EC_USE_COMPLEMENT="0",
            EC_USE_PHI="0",
            EC_USE_CHAIN_COVERAGE="0",
        ):
            pruner_no_dep = ECPruner(debug=False)
            probs_no_dep, _, prob_info_no_dep = pruner_no_dep.compute_qfid_probability(
                576,
                device,
                b=b,
                semantic_response_source="clip",
            )
        assert prob_info_no_dep["mode"] == "fixed"
        assert abs(prob_info_no_dep["adaptive_depolarize"] - 0.0) <= 1e-8
        assert torch.max(torch.abs(probs - probs_no_dep)).item() > 1e-8


def test_qfid_adaptive_depolarize_probability(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE_MODE="adaptive",
        EC_QFID_DEPOLARIZE_MIN="0.05",
        EC_QFID_DEPOLARIZE_MAX="0.25",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        b = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=b,
            semantic_response_source="clip",
        )
        assert source_used == "semantic"
        assert probs.shape == (576,)
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert prob_info["mode"] == "adaptive"
        assert 0.0 <= prob_info["entropy_norm"] <= 1.0
        assert 0.05 <= prob_info["adaptive_depolarize"] <= 0.25
        assert prob_info["p_min"] >= 0.0
        assert prob_info["p_max"] >= prob_info["p_min"]


def test_qfid_anchor_none_matches_pure_qf(ECPruner, device):
    visual_tokens = torch.randn(1, 576, 128, device=device)
    semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_none = ECPruner(debug=False)
        keep_none, debug_none = pruner_none.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_pure = ECPruner(debug=False)
        keep_pure, debug_pure = pruner_pure.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    assert torch.equal(keep_none, keep_pure)
    assert debug_none["qfid_info"]["anchor_mode"] == "none"
    assert debug_none["qfid_info"]["anchor_num"] == 0
    assert debug_none["qfid_info"]["qf_fill_count"] == 64
    assert debug_pure["qfid_info"]["anchor_num"] == 0


def test_qfid_measurement_anchor_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_ANCHOR_MODE="measurement",
        EC_QFID_MEASURE_ANCHOR_RATIO="0.125",
        EC_QFID_MEASURE_ANCHOR_MIN="0",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["anchor_mode"] == "measurement"
        assert debug_info["qfid_info"]["anchor_num"] == 8
        assert debug_info["qfid_info"]["qf_fill_count"] == 56
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert debug_info["qfid_info"]["anchor_prob_max"] >= debug_info["qfid_info"]["anchor_prob_min"]
        assert debug_info["qfid_info"]["anchor_prob_mean"] >= debug_info["qfid_info"]["anchor_prob_min"]
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_softprior_none_matches_pure_qf(ECPruner, device):
    visual_tokens = torch.randn(1, 576, 128, device=device)
    semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_MEASURE_PRIOR_LAMBDA="0.20",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_none = ECPruner(debug=False)
        keep_none, debug_none = pruner_none.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_pure = ECPruner(debug=False)
        keep_pure, _ = pruner_pure.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    assert torch.equal(keep_none, keep_pure)
    assert debug_none["qfid_info"]["measure_prior_mode"] == "none"
    assert abs(debug_none["qfid_info"]["measure_prior_lambda"] - 0.20) <= 1e-8


def test_qfid_softprior_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_MEASURE_PRIOR_MODE="soft",
        EC_QFID_MEASURE_PRIOR_LAMBDA="0.10",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["measure_prior_mode"] == "soft"
        assert abs(debug_info["qfid_info"]["measure_prior_lambda"] - 0.10) <= 1e-8
        assert 0.0 <= debug_info["qfid_info"]["p_norm_min"] <= 1.0
        assert 0.0 <= debug_info["qfid_info"]["p_norm_max"] <= 1.0
        assert debug_info["qfid_info"]["score_max"] >= debug_info["qfid_info"]["score_min"]
        assert debug_info["qfid_info"]["final_keep_size"] == 64
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_measurement_probability(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="measurement",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        b = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=b,
            semantic_response_source="clip",
        )
        assert source_used == "measurement"
        assert probs.shape == (576,)
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert prob_info["source_fallback"] is False
        assert prob_info["measurement_alpha_mode"] == "uniform_alpha"
        assert prob_info["measurement_score_max"] >= prob_info["measurement_score_min"]
        assert prob_info["measurement_prob_max"] >= prob_info["measurement_prob_min"]


def test_qfid_measurement_probability_fallback(ECPruner, device):
    with env_override(
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="measurement",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        b = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        probs, source_used, prob_info = pruner.compute_qfid_probability(
            576,
            device,
            b=b,
            semantic_response_source="uniform",
        )
        assert source_used in {"semantic_fallback", "uniform"}
        assert probs.shape == (576,)
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum().item()) - 1.0) <= 1e-5
        assert prob_info["source_fallback"] is True


def test_qfid_measurement_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="measurement",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["prob_source"] == "measurement"
        assert debug_info["qfid_info"]["measurement_score_max"] >= debug_info["qfid_info"]["measurement_score_min"]
        assert debug_info["qfid_info"]["measurement_prob_max"] >= debug_info["qfid_info"]["measurement_prob_min"]
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_spatial_state_none_matches_pure_qf(ECPruner, device):
    visual_tokens = torch.randn(1, 576, 128, device=device)
    semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="0",
        EC_QFID_SPATIAL_LAMBDA="0.30",
        EC_QFID_SPATIAL_SIGMA="0.20",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_none = ECPruner(debug=False)
        keep_none, debug_none = pruner_none.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner_pure = ECPruner(debug=False)
        keep_pure, _ = pruner_pure.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )

    assert torch.equal(keep_none, keep_pure)
    assert debug_none["qfid_info"]["spatial_state_enabled"] is False
    assert debug_none["qfid_info"]["spatial_state_applied"] is False


def test_qfid_spatial_state_select(ECPruner, device):
    with env_override(
        EC_SOLVER="greedy",
        EC_SCORE_SOURCE="qfid",
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="1",
        EC_QFID_SPATIAL_LAMBDA="0.10",
        EC_QFID_SPATIAL_SIGMA="0.20",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
        EC_USE_SEMANTIC_CANDIDATE="0",
        EC_USE_SPATIAL_CANDIDATE="0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 576, 128, device=device)
        semantic_response = torch.softmax(torch.randn(576, 4, device=device), dim=-1)
        keep_idx, debug_info = pruner.select(
            visual_tokens,
            keep_num=64,
            question="what is on the table",
            b=semantic_response,
        )
        assert_valid_keep_idx(keep_idx, 64, 576, device)
        assert debug_info["qfid_info"]["spatial_state_enabled"] is True
        assert debug_info["qfid_info"]["spatial_state_applied"] is True
        assert debug_info["qfid_info"]["spatial_state_fallback"] is False
        assert debug_info["qfid_info"]["grid_size"] == 24
        assert debug_info["qfid_info"]["spatial_factor_max"] >= debug_info["qfid_info"]["spatial_factor_min"]
        assert debug_info["qfid_info"]["spatial_factor_mean"] >= debug_info["qfid_info"]["spatial_factor_min"]
        assert_finite_debug_values(debug_info["qfid_info"])


def test_qfid_spatial_state_fallback_nonsquare(ECPruner, device):
    with env_override(
        EC_QFID_TAU="0.50",
        EC_QFID_EPS="1e-6",
        EC_QFID_PROB_SOURCE="semantic",
        EC_QFID_KERNEL="density",
        EC_QFID_DEPOLARIZE="0.15",
        EC_QFID_SPATIAL_STATE="1",
        EC_QFID_SPATIAL_LAMBDA="0.10",
        EC_QFID_SPATIAL_SIGMA="0.20",
        EC_QFID_MEASURE_PRIOR_MODE="none",
        EC_QFID_ANCHOR_MODE="none",
    ):
        pruner = ECPruner(debug=False)
        visual_tokens = torch.randn(1, 500, 64, device=device)
        semantic_response = torch.softmax(torch.randn(500, 4, device=device), dim=-1)
        keep_idx, qfid_info = pruner.select_by_quantum_fidelity(
            visual_tokens,
            K=32,
            semantic_response=semantic_response,
            semantic_response_source="clip",
        )
        assert_valid_keep_idx(keep_idx, 32, 500, device)
        assert qfid_info["spatial_state_enabled"] is True
        assert qfid_info["spatial_state_applied"] is False
        assert qfid_info["spatial_state_fallback"] is True
        assert qfid_info["grid_size"] == 0


def test_energy_refinement_not_worse(ECPruner, device):
    with env_override(
        EC_SOLVER="qanneal",
        EC_QA_STEPS="30",
        EC_QA_T0="1.0",
        EC_QA_TEND="0.01",
        EC_QA_SEED="42",
        EC_QA_MAX_SWAP_RATIO="0.15",
        EC_LAMBDA="0.1",
        EC_DELTA="0.1",
        EC_USE_CHAIN_COVERAGE="0",
    ):
        pruner = ECPruner(debug=False)
        candidate_size = 48
        keep_num = 16
        a_c = torch.rand(candidate_size, device=device)
        base = torch.rand(candidate_size, candidate_size, device=device)
        R_c = (base + base.t()) / 2.0
        C_c = torch.flip(R_c, dims=[0])
        R_c.fill_diagonal_(0.0)
        C_c.fill_diagonal_(0.0)
        greedy_local_idx = pruner.greedy_select(a_c, R_c, C_c, keep_num)
        greedy_mask = pruner._selected_to_mask(greedy_local_idx, candidate_size, device)
        greedy_energy = float(
            pruner.compute_energy(
                greedy_mask,
                a_c,
                R_c,
                C_c,
                pruner.lambda_repulsion,
                pruner.delta_complement,
            ).item()
        )
        keep_idx, debug_info = pruner.qanneal_select(
            a_c=a_c,
            R_c=R_c,
            C_c=C_c,
            K=keep_num,
            candidate_idx=torch.arange(candidate_size, device=device),
            init_selected=greedy_mask,
            lambda_coef=pruner.lambda_repulsion,
            delta_coef=pruner.delta_complement,
        )
        assert keep_idx.numel() == keep_num
        assert torch.unique(keep_idx).numel() == keep_num
        assert debug_info["best_energy"] <= greedy_energy + 1e-6


def test_chain_coverage_energy_rewards_evidence_completion(ECPruner, device):
    with env_override(
        EC_USE_CHAIN_COVERAGE="1",
        EC_GAMMA_CHAIN="1.0",
        EC_USE_REPULSION="0",
        EC_USE_COMPLEMENT="0",
        EC_USE_PHI="0",
    ):
        pruner = ECPruner(debug=False)
        a_c = torch.tensor([1.0, 0.9, 0.8], device=device)
        zeros = torch.zeros(3, 3, dtype=torch.float32, device=device)
        b_c = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        relation = pruner._prepare_chain_relation_matrix(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        greedy_idx = pruner.greedy_select(
            a_c,
            zeros,
            zeros,
            keep_num=2,
            b_c=b_c,
            chain_relation_matrix=relation,
            gamma_chain=1.0,
        )
        assert set(greedy_idx.tolist()) == {0, 2}

        same_end_mask = pruner._selected_to_mask(torch.tensor([0, 1], device=device), 3, device)
        completed_mask = pruner._selected_to_mask(torch.tensor([0, 2], device=device), 3, device)
        same_end_energy = float(
            pruner.compute_energy(
                same_end_mask,
                a_c,
                zeros,
                zeros,
                0.0,
                0.0,
                b_c=b_c,
                chain_relation_matrix=relation,
                gamma_chain=1.0,
            ).item()
        )
        completed_energy = float(
            pruner.compute_energy(
                completed_mask,
                a_c,
                zeros,
                zeros,
                0.0,
                0.0,
                b_c=b_c,
                chain_relation_matrix=relation,
                gamma_chain=1.0,
            ).item()
        )
        assert completed_energy < same_end_energy - 1e-6


def main():
    ECPruner = load_ec_pruner_class()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "--qmo-cr-only" in sys.argv:
        test_qmo_cr_selector_function(device)
        test_qfid_qmo_cr_select(ECPruner, device)
        print(f"QMO-CR tests passed on {device.type}")
        return
    if "--qmo-gated-only" in sys.argv:
        test_qmo_gated_selector_function(device)
        test_qfid_qmo_gated_select(ECPruner, device)
        print(f"QMO-gated tests passed on {device.type}")
        return
    if "--qsp-only" in sys.argv:
        test_qsp_density_projective_selector_function(device)
        test_qfid_qsp_density_projective_select(ECPruner, device)
        test_qfid_qfi_core_complement_recovery_select(ECPruner, device)
        test_run_gqa_profile_contains_qsp_density_projective_profiles()
        print(f"QSP density-projective tests passed on {device.type}")
        return
    if "--qfi-cards-only" in sys.argv:
        test_qfi_cards_selector_functions(device)
        test_qfid_qfi_cards_select(ECPruner, device)
        test_run_gqa_profile_contains_qfi_cards_profiles()
        print(f"QFI card tests passed on {device.type}")
        return

    test_greedy_solver_fixed_k(ECPruner, device)
    test_qanneal_solver_fixed_k(ECPruner, device)
    test_candidate_padding_to_k(ECPruner, device)
    test_env_bool_parsing(ECPruner)
    test_projection_overlap_kernel_modes(device)
    test_pairwise_matrices_are_symmetric(ECPruner, device)
    test_qmo_score_and_select(ECPruner, device)
    test_qfid_amplitude_select(ECPruner, device)
    test_qfid_density_select(ECPruner, device)
    test_qfid_clsmix_probability_and_select(ECPruner, device)
    test_qfid_clsmix_geometric_probability_and_select(ECPruner, device)
    test_qfid_clsmix_fallback_to_semantic(ECPruner, device)
    test_qfid_clsmix_gate_disabled_matches_legacy(ECPruner, device)
    test_qfid_clsmix_agreement_gate_select(ECPruner, device)
    test_qfid_cls_gate_does_not_affect_semantic_source(ECPruner, device)
    test_qfid_cls_only_qf_select(ECPruner, device)
    test_qfid_cls_topk_select(ECPruner, device)
    test_qfid_semantic_topk_select(ECPruner, device)
    test_qfid_uniform_density_select(ECPruner, device)
    test_evidence_recovery_selector_function(device)
    test_core_then_recover_selector_function(device)
    test_adaptive_core_recover_selector_function(device)
    test_qficr_structural_ablation_controls(ECPruner, device)
    test_qmo_cr_selector_function(device)
    test_qmo_gated_selector_function(device)
    test_qsp_density_projective_selector_function(device)
    test_qfid_selector_default_and_explicit_residual_match(ECPruner, device)
    test_qfid_evidence_recovery_select(ECPruner, device)
    test_qfid_core_then_recover_select(ECPruner, device)
    test_qfid_adaptive_core_recover_select(ECPruner, device)
    test_qfid_qmo_cr_select(ECPruner, device)
    test_qfid_qmo_gated_select(ECPruner, device)
    test_qfid_qsp_density_projective_select(ECPruner, device)
    test_qfid_qfi_core_complement_recovery_select(ECPruner, device)
    test_qfi_cards_selector_functions(device)
    test_qfid_qfi_cards_select(ECPruner, device)
    test_qfid_visual_kcenter_select(ECPruner, device)
    test_qfid_gate_sidecar(ECPruner, device)
    test_qfid_budget_calibration_function(device)
    test_qfid_budget_calibration_disabled_matches_legacy(ECPruner, device)
    test_qfid_budget_calibration_select_and_sidecar(ECPruner, device)
    test_qfid_cls_entropy_weighted_reduce(ECPruner, device)
    test_clip_cls_attention_layer_reconstruction(device)
    test_run_gqa_profile_contains_spectral_profiles()
    test_run_gqa_profile_contains_evidence_recover_profile()
    test_run_gqa_profile_contains_core_then_recover_profiles()
    test_run_gqa_profile_contains_adaptive_recover_profiles()
    test_run_gqa_profile_contains_qmo_gated_profiles()
    test_run_gqa_profile_contains_qficr_ablation_profiles()
    test_run_gqa_profile_contains_qsp_density_projective_profiles()
    test_run_gqa_profile_contains_qfi_cards_profiles()
    test_qfid_depolarize_probability(ECPruner, device)
    test_qfid_adaptive_depolarize_probability(ECPruner, device)
    test_qfid_anchor_none_matches_pure_qf(ECPruner, device)
    test_qfid_measurement_anchor_select(ECPruner, device)
    test_qfid_softprior_none_matches_pure_qf(ECPruner, device)
    test_qfid_softprior_select(ECPruner, device)
    test_qfid_measurement_probability(ECPruner, device)
    test_qfid_measurement_probability_fallback(ECPruner, device)
    test_qfid_measurement_select(ECPruner, device)
    test_qfid_spatial_state_none_matches_pure_qf(ECPruner, device)
    test_qfid_spatial_state_select(ECPruner, device)
    test_qfid_spatial_state_fallback_nonsquare(ECPruner, device)
    test_energy_refinement_not_worse(ECPruner, device)
    test_chain_coverage_energy_rewards_evidence_completion(ECPruner, device)

    print(f"EC-Pruner smoke tests passed on {device.type}")


if __name__ == "__main__":
    main()
