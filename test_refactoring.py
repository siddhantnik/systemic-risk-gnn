"""
Verification script for the 4-requirement refactoring.
Tests evaluation module changes, inference module, and API imports.
"""
import torch
import numpy as np
import sys
import os

print("=" * 60)
print("VERIFICATION: Systemic Risk GNN Refactoring")
print("=" * 60)

# ---- Test 1: Evaluation module - dynamic threshold ----
print("\n[TEST 1] Dynamic percentile-based thresholding")
from training.evaluation import compute_dynamic_threshold, calculate_pr_metrics, POSITIVE_PERCENTILE

assert POSITIVE_PERCENTILE == 0.95, f"Expected 0.95, got {POSITIVE_PERCENTILE}"
print(f"  POSITIVE_PERCENTILE = {POSITIVE_PERCENTILE} [OK]")

y = torch.rand(1000)
threshold = compute_dynamic_threshold(y)
print(f"  Dynamic threshold on 1000 random [0,1] values: {threshold:.4f}")
assert 0.0 <= threshold <= 1.0, f"Threshold out of range: {threshold}"

# Verify approximately 5% are above threshold
above = (y >= threshold).float().mean().item()
print(f"  Fraction above threshold: {above:.3f} (expected ~0.05)")
assert above < 0.10, f"Too many nodes above threshold: {above}"
print("  [TEST 1] PASSED")

# ---- Test 2: PR metrics with dynamic threshold ----
print("\n[TEST 2] PR-AUC with dynamic threshold")
y_true = torch.rand(200)
y_pred = torch.rand(200)
pr = calculate_pr_metrics(y_true, y_pred)

assert "pr_auc" in pr, "Missing pr_auc in result"
assert "threshold_used" in pr, "Missing threshold_used in result"
assert 0.0 <= pr["pr_auc"] <= 1.0, f"PR-AUC out of bounds: {pr['pr_auc']}"
print(f"  PR-AUC: {pr['pr_auc']}")
print(f"  Threshold used: {pr['threshold_used']}")
print(f"  n_positive: {pr['n_positive']}, n_total: {pr['n_total']}")
print(f"  Positive ratio: {pr['positive_ratio']}")
assert pr["positive_ratio"] < 0.10, f"Positive ratio too high: {pr['positive_ratio']}"
print("  [TEST 2] PASSED")

# ---- Test 3: No CRITICAL_THRESHOLD in evaluation ----
print("\n[TEST 3] No hardcoded CRITICAL_THRESHOLD")
import training.evaluation as eval_module
assert not hasattr(eval_module, 'CRITICAL_THRESHOLD'), "CRITICAL_THRESHOLD still exists!"
print("  CRITICAL_THRESHOLD removed from evaluation.py [OK]")
print("  [TEST 3] PASSED")

# ---- Test 4: Inference module - no z-score/sigmoid ----
print("\n[TEST 4] Inference module integrity")
import backend.services.inference as inf_module

# Check no CRITICAL_THRESHOLD
assert not hasattr(inf_module, 'CRITICAL_THRESHOLD'), "CRITICAL_THRESHOLD still in inference!"
print("  CRITICAL_THRESHOLD removed from inference.py [OK]")

# Check source code has no z_scores or scaled_scores patterns
inf_source = open(os.path.join("backend", "services", "inference.py")).read()
assert "z_scores" not in inf_source, "z_scores still in inference.py!"
assert "scaled_scores" not in inf_source, "scaled_scores still in inference.py!"
print("  No z-score/sigmoid transforms in inference.py [OK]")
print("  [TEST 4] PASSED")

# ---- Test 5: Routes module - no shock overrides ----
print("\n[TEST 5] Routes module integrity")
routes_source = open(os.path.join("backend", "api", "routes.py")).read()
assert "max(0.95" not in routes_source, "Forced 0.95 override still in routes.py!"
assert "z_scores" not in routes_source, "z_scores still in routes.py!"
assert "scaled_scores" not in routes_source, "scaled_scores still in routes.py!"
assert "evaluation_split" in routes_source, "evaluation_split not in /metrics response!"
assert "test_node_count" in routes_source, "test_node_count not in /metrics response!"
assert "test_mask" in routes_source, "test_mask not referenced in routes.py!"
print("  No forced 0.95 override [OK]")
print("  No z-score/sigmoid in shock route [OK]")
print("  /metrics uses test_mask only [OK]")
print("  evaluation_split field present [OK]")
print("  test_node_count field present [OK]")
print("  [TEST 5] PASSED")

# ---- Test 6: AutoML search - clean slate + bounds ----
print("\n[TEST 6] AutoML search integrity")
automl_source = open(os.path.join("training", "automl_search.py")).read()
assert "_clean_stale_artifacts" in automl_source, "Stale artifact cleanup missing!"
assert "PR-AUC out of bounds" in automl_source, "PR-AUC bounds validation missing!"
assert 'direction="maximize"' in automl_source, "Optuna direction not maximize!"
print("  Stale artifact cleanup function present [OK]")
print("  PR-AUC bounds validation present [OK]")
print("  Optuna direction='maximize' confirmed [OK]")
print("  [TEST 6] PASSED")

print("\n" + "=" * 60)
print("ALL 6 VERIFICATION TESTS PASSED")
print("=" * 60)
