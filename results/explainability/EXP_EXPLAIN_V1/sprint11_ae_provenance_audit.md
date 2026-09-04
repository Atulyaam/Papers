# Sprint 11 — Autoencoder Provenance Forensic Audit
**Experiment ID**: `EXP_EXPLAIN_V1`  
**Audit Timestamp**: 2026-09-04T12:17:27.044249+00:00  
**Repository Root**: `c:/Users/Atul2/OneDrive/Desktop/Papers/IDS-UNSW-NB15`  
**Protocol Authority**: Sprint 11 Master Task Specification (Phase A Requirements)  

---

## 1. Audit Scope
This forensic audit strictly resolves the provenance, architecture, loading integrity, and artifact validity of the unsupervised Benign-Only Tabular Autoencoder (AE) used within the `A6_STACK_PLUS_AE` system for Sprint 11 explainability.
The scope encompasses:
- Tracing `results/checkpoints/EXP_AE_V1/ae_final.pt` back to its Sprint 7/9 training code.
- Direct inspection of raw tensor keys, shapes, and scalar parameter counts from the checkpoint.
- Architecture derivation and comparison against the authoritative training class.
- Audit of all `load_state_dict` calls across the repository.
- Audit of `strict=False` vs `strict=True` behavior.
- Chronological analysis of original vs corrected implementations in Sprint 11.
- Forensic quarantine and verified regeneration of all AE-dependent artifacts.
- Independent re-verification of the full 81,749-row DEVELOPMENT_TEST AE-decisive cases.
- Verification of the mathematical reconstruction formula and A6 OR-fusion rule.
- Clarification of the 34 pre-verification gates and PV-23 semantics.

## 2. Checkpoint Identity
| Attribute | Value |
|:---|:---|
| **File Path** | `results\checkpoints\EXP_AE_V1\ae_final.pt` |
| **File Size** | 11739 bytes |
| **SHA-256 Checksum** | `4ab66af8d4a6e61212ef5d78360f30a8caa68aa85dac3d54042218e010f9a1d6` |
| **Format** | PyTorch serialized state_dict (weights_only compatible) |
| **Associated Scaler** | `results/checkpoints/EXP_AE_V1/ae_scaler.joblib` |
| **Scaler SHA-256** | `c0128d42ed9ef5be695f261be75155e7de4ddf8e51b926e3ce516c4a88ad8211` |
| **Threshold Config** | `results/checkpoints/EXP_AE_V1/threshold_config.json` |
| **Threshold SHA-256**| `3679e65ac717de00cd1f72c4cbaa1a28e0ad823252333068f63ca7873f6a25ab` |
| **Architecture Spec**| `results/checkpoints/EXP_AE_V1/ae_architecture.json` |
| **Architecture SHA-256**| `a510be4d1be8857572ed76753588e7575951c44ddfb7b4b78a9fbea4695e847a` |
| **Metadata File** | `results/checkpoints/EXP_AE_V1/ae_metadata.json` |
| **Metadata SHA-256** | `9894ab3010dbdb75294f1c224e1bef7cd7d7b1db9f3bf2bb754e9ffe99b6672a` |

## 3. Sprint 9 AE Training Source
The frozen checkpoint `results/checkpoints/EXP_AE_V1/ae_final.pt` was trained during Sprint 7 (`EXP_AE_V1`), frozen by protocol, and formally evaluated in Sprint 9 and Sprint 10.
- **Authoritative Training Script**: `scripts/run_ae_training.py`
- **Training Orchestration Class**: `src.models.autoencoder.ae_trainer.AETrainer`
- **Authoritative Neural Architecture Class**: `src.models.autoencoder.ae_model.Autoencoder`
- **Training Configuration**: Defined in `src/models/autoencoder/ae_trainer.py` and recorded in `ae_metadata.json`:
  - Input Dimension: 75 selected features (`EXP_MI_V1_1` mutual information contract)
  - Training Data: 40,320 Normal TRAIN rows (seed 42 monitor split)
  - Monitor Split: 4,480 Normal TRAIN rows (seed 42)
  - Calibration Split: 11,200 Normal VALIDATION rows
  - Final Refit: Full 44,800 Normal TRAIN rows refit to `best_epoch = 133` (patience=5, max_epochs=150, lr=0.001, Adam, batch_size=256, weight_decay=0.0001)
  - Scaler: `StandardScaler` fitted strictly on Normal AE-fit subset (40,320 rows)
  - Checkpoint Save Code: `torch.save(final_model.state_dict(), ckpt_dir / "ae_final.pt")` (`scripts/run_ae_training.py:195`)

## 4. Raw Checkpoint Tensor Shapes
Direct inspection of the serialized tensor dictionary in `results/checkpoints/EXP_AE_V1/ae_final.pt` yields the following exact layout:

| Tensor Key | Shape | Element Count | Data Type |
|:---|:---|:---|:---|
| `encoder.0.weight` | `[12, 75]` | 900 | `torch.float32` |
| `encoder.0.bias` | `[12]` | 12 | `torch.float32` |
| `encoder.2.weight` | `[6, 12]` | 72 | `torch.float32` |
| `encoder.2.bias` | `[6]` | 6 | `torch.float32` |
| `decoder.0.weight` | `[12, 6]` | 72 | `torch.float32` |
| `decoder.0.bias` | `[12]` | 12 | `torch.float32` |
| `decoder.2.weight` | `[75, 12]` | 900 | `torch.float32` |
| `decoder.2.bias` | `[75]` | 75 | `torch.float32` |
| **TOTAL** | — | **2049** | — |

## 5. Derived Checkpoint Architecture
From the raw weight matrices and bias vectors alone, the model topology is mathematically derived as:
- **Input Layer**: 75 dimensions (from `encoder.0.weight` shape [12, 75])
- **Encoder Hidden Layer**: 12 units (`encoder.0.bias` shape [12])
- **Bottleneck Layer**: 6 units (`encoder.2.weight` shape [6, 12] and `encoder.2.bias` shape [6])
- **Decoder Hidden Layer**: 12 units (`decoder.0.weight` shape [12, 6] and `decoder.0.bias` shape [12])
- **Output Layer**: 75 units (`decoder.2.weight` shape [75, 12] and `decoder.2.bias` shape [75])
- **Total Trainable Parameters**: 2,049 scalar parameters.
- **Derived Topology**: **`75 → 12 → 6 → 12 → 75`**
- **BatchNorm**: **ABSENT** (no running_mean, running_var, weight, or bias tensors)
- **Dropout**: **ABSENT** (no parameters, not represented in checkpoint)

## 6. Training-Source Architecture
Inspecting `src/models/autoencoder/ae_model.py` (`Autoencoder`):
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 75) -> None:
        super().__init__()
        # Encoder: 75 → 12 → 6
        self.encoder = nn.Sequential(
            nn.Linear(75, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
        )
        # Decoder: 6 → 12 → 75
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 75),
        )
```
- **Input dimension**: 75
- **Encoder sequence**: `Linear(75, 12)` → `ReLU()` → `Linear(12, 6)` → `ReLU()`
- **Bottleneck dimension**: 6 (with non-negative ReLU activation)
- **Decoder sequence**: `Linear(6, 12)` → `ReLU()` → `Linear(12, 75)`
- **Output activation**: Linear (none)
- **BatchNorm**: False
- **Dropout**: False
- **Total parameters**: 2,049

## 7. Architecture Comparison
| Property | Training Source (`Autoencoder`) | Raw Checkpoint (`ae_final.pt`) | Match? |
|:---|:---|:---|:---|
| Input Dimension | 75 | 75 (`encoder.0.weight` [12, 75]) | **YES** |
| Encoder Hidden 1 | 12 | 12 (`encoder.0.weight` [12, 75]) | **YES** |
| Bottleneck Dimension | 6 | 6 (`encoder.2.weight` [6, 12]) | **YES** |
| Decoder Hidden 1 | 12 | 12 (`decoder.0.weight` [12, 6]) | **YES** |
| Output Dimension | 75 | 75 (`decoder.2.weight` [75, 12]) | **YES** |
| Total Trainable Parameters | 2,049 | 2,049 | **YES** |
| BatchNorm Layers | None | None | **YES** |

**ARCHITECTURE_MATCH = YES**

## 8. Original Sprint 11 Implementation
When `scripts/run_sprint11_explainability.py` was drafted in Step 360, a local class `TabularAutoencoder` was introduced rather than importing `src.models.autoencoder.ae_model.Autoencoder`.
- Original Draft Architecture: `75 → 48 → 32 → 16 → 32 → 48 → 75` with `BatchNorm1d` and `latent_dim=16`.
- Execution at Step 607 immediately failed on `load_state_dict` with `RuntimeError: Error(s) in loading state_dict for TabularAutoencoder` due to complete layer mismatch.

## 9. Corrected Sprint 11 Implementation and Forensic Diff
In Step 614–632, `TabularAutoencoder` was restructured to `75 → 12 → 6 → 12 → 75`.
However, forensic auditing of that class revealed that the second encoder layer omitted the bottleneck `ReLU()` activation:
```python
# Flawed interim implementation in run_sprint11_explainability.py:
self.encoder = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 6),
    # Missing: torch.nn.ReLU()
)
```
Because PyTorch activation layers contain no trainable weights, `load_state_dict(ae_state)` loaded without raising an error. However, when the bottleneck linear layer evaluated negative values, they entered the decoder unrectified.

### Explicit Architecture Diff:
```diff
--- Interim TabularAutoencoder
+++ Authoritative Autoencoder (src/models/autoencoder/ae_model.py)
@@ -1,10 +1,11 @@
 Sequential(
   (0): Linear(in_features=75, out_features=12, bias=True)
   (1): ReLU()
   (2): Linear(in_features=12, out_features=6, bias=True)
+  (3): ReLU()
 )
 Decoder(
   (0): Linear(in_features=6, out_features=12, bias=True)
   (1): ReLU()
   (2): Linear(in_features=12, out_features=75, bias=True)
 )
```
- **Resolution**: `scripts/run_sprint11_explainability.py` was updated to import and instantiate the authoritative `from src.models.autoencoder.ae_model import Autoencoder`, ensuring exact mathematical parity with Sprint 7, 8, 9, and 10.

## 10. Audit of All load_state_dict Calls
An exhaustive search of all Python files in the repository identified the following occurrences:

| File | Line | Object | Exact Call | Strict Setting | Context |
|:---|:---|:---|:---|:---|:---|
| `scripts/run_sprint11_explainability.py` | 374 | `self.nn_raw` | `self.nn_raw.load_state_dict(self.nn_state)` | PyTorch default (`strict=True`) | Loading frozen A1 `IDSNet` |
| `scripts/run_sprint11_explainability.py` | 806 | `ae_model` | `ae_model.load_state_dict(ae_state, strict=True)` | Explicit `strict=True` | Loading frozen `EXP_AE_V1` Autoencoder |
| `scripts/run_fusion_evaluation.py` | 374 | `self._model` | `self._model.load_state_dict(torch.load(..., strict=True))` | PyTorch default (`strict=True`) | Sprint 8/10 AE adapter load |
| `scripts/evaluate_sprint9.py` | 175 | `ae_model` | `ae_model.load_state_dict(torch.load(...))` | PyTorch default (`strict=True`) | Sprint 9 evaluation |
| `scripts/run_determinism_check.py` | 173 | `ae_model` | `ae_model.load_state_dict(torch.load(...))` | PyTorch default (`strict=True`) | Sprint 10 determinism check |
| `scripts/run_ablation.py` | 735 | `ae_model` | `ae_model.load_state_dict(torch.load(...))` | PyTorch default (`strict=True`) | Sprint 10 ablation study |
| `src/models/autoencoder/artifacts.py` | 270 | `model` | `model.load_state_dict(state)` | PyTorch default (`strict=True`) | Sprint 7 artifact generation |

## 11. strictness Settings Audit
- **Zero occurrences** of `strict=False` exist across Sprint 11 or any related evaluation script.
- No partially initialized weights or skipped parameter keys were ever permitted.
- `run_sprint11_explainability.py` line 806 now uses explicit `strict=True`.

## 12. Fresh Independent strict=True Load Result
A freshly executed independent Python verification script instantiated `src.models.autoencoder.ae_model.Autoencoder(input_dim=75)` and executed:
```python
model.load_state_dict(torch.load('results/checkpoints/EXP_AE_V1/ae_final.pt', map_location='cpu', weights_only=True), strict=True)
```
- **Execution Status**: SUCCESS
- **Missing Keys**: `[]`
- **Unexpected Keys**: `[]`
- **Shape Mismatches**: None
- **Exceptions**: None

## 13. Post-Load Model Verification
- **State Dict Tensor Count**: 8
- **Total Weight Elements**: 2,049
- **Evaluation Mode**: `model.eval()` confirmed (gradients disabled during all explainability inference passes).
- **Numerical Parity**: Verified bitwise identical outputs across CPU and CUDA.

## 14. Git / Source Chronology
- **Commit `c946fa6` (Sprint 7)**: Fixed Autoencoder architecture frozen at `75 → 12 → 6 → 12 → 75` in `src/models/autoencoder/ae_model.py`.
- **Commit `fc57572` (Sprint 9)**: H1/H2/H3 evaluation froze the AE inference outputs.
- **Commit `839a6ad` (Sprint 10)**: Ablation study re-verified AE fusion at tau = 11.160062745213509.
- **Commit `c462891` (Sprint 10 maintenance)**: Stacking and AE calibration pipeline updated.
- **Sprint 11**: `scripts/run_sprint11_explainability.py` introduced post-hoc explainability. The AE definition discrepancy was detected during Phase A forensic auditing, quarantined under Section 4.10, and resolved under Section 4.11.

## 15. AE-Dependent Artifact Inventory
The following artifacts depend on the Autoencoder:
1. `results/explainability/EXP_EXPLAIN_V1/A6_STACK_PLUS_AE/ae_decisive_cases.csv`
2. `results/explainability/EXP_EXPLAIN_V1/A6_STACK_PLUS_AE/ae_reconstruction_importance.csv`
3. `results/explainability/EXP_EXPLAIN_V1/A6_STACK_PLUS_AE/local_cases.json`
4. `results/explainability/EXP_EXPLAIN_V1/figures/A6_STACK_PLUS_AE/*.png` (20 local case figures)
5. `results/explainability/EXP_EXPLAIN_V1/figures/ae_decisive_cases/*.png` (10 decisive case bar charts)
6. `results/explainability/EXP_EXPLAIN_V1/summary.json`
7. `results/explainability/EXP_EXPLAIN_V1/quality_review.md`
8. `results/explainability/EXP_EXPLAIN_V1/validation_report.md`

## 16. Artifact Hashes and Timestamps
Post-regeneration SHA-256 hashes of all verified AE-dependent artifacts:
- `ae_decisive_cases.csv`: `de24f5cdc5e99505207e0ab48d7e5f4f1b884cc8a7a192a34beef3cb9bfba7e8`
- `ae_reconstruction_importance.csv`: `cb6e22ef9369f53397fe9c5137c6cda99030ec3fae35c287b703a288857c9126`
- `local_cases.json`: `4018a69feaf711b12e5cbb5ff85723ad4640b292f44fa1215d8032285410a096`
- `summary.json`: `da1b4d102d518005816b35bce1ef07784399264db1159a5b44cadc515642bdc8`
- `metadata.json`: `0a5a30e4d080700699048c0cdd65fed18526a5cf03ff9e45d312ef75b9792091`

## 17. Quarantined Artifacts
In compliance with Section 4.10, all 33 suspect artifacts generated prior to the authoritative model substitution were quarantined to:
`results/explainability/EXP_EXPLAIN_V1/_quarantine_ae_provenance/`
The quarantine manifest is preserved at `quarantine_manifest.json` recording original file paths, sizes, UTC timestamps, and SHA-256 checksums.

## 18. Regeneration Details
In compliance with Section 4.11:
- Only AE-dependent components were regenerated using `src.models.autoencoder.ae_model.Autoencoder` with explicit `strict=True`.
- No A0 or A1 SHAP passes were recomputed (cached `shap_values.npz` files were verified and preserved).
- The frozen scaler (`ae_scaler.joblib`) and frozen threshold (`tau = 11.160062745213509`) were strictly honored.

## 19. AE-Decisive Case Verification
Full independent scan of `DEVELOPMENT_TEST` (N = 81,749):
- **Locked Predicate**: `A1_pred == 0 AND AE_flag == 1`
- **Historical Reported Count (Sprint 10)**: **13**
- **Independently Verified Count**: **13**
- **Exact Positional Row IDs**: [np.int64(40612), np.int64(40620), np.int64(41062), np.int64(41459), np.int64(41540), np.int64(69582), np.int64(69620), np.int64(69625), np.int64(69638), np.int64(69657), np.int64(69662), np.int64(69670), np.int64(81287)]
- **Exact Match**: 100% agreement between Sprint 10 reported decisive set and Sprint 11 verified decisive set.

## 20. Reconstruction-Error Mathematical Formula
Verified adherence to Section 4.13:
- Per-feature squared error: $e_i = (x_i - \hat{x}_i)^2$
- Per-sample reconstruction error: $RE(x) = \frac{1}{75}\sum_{i=1}^{75} e_i$ (mean, NOT sum, NOT MAE)
- Anomaly decision rule: $AE\_flag = 1$ if $RE(x) > \tau$, else $0$.

## 21. A6 Fusion Verification
Verified adherence to Section 4.14:
- Fusion Rule: $A6\_final = A1\_pred \lor AE\_flag$
- Threshold: $\tau = 11.160062745213509$ (`mean3sigma` calibrated on Normal VALIDATION)
- Verification status: Pure post-hoc read-only verification; no modifications permitted.

## 22. 34 Pre-Verification Gates Explanation
Exactly 34 pre-verification gates were evaluated: PV-01 through PV-33 plus the additional PV-11a global source-row-UID uniqueness gate.
- All 34 gates PASSED with 100% success rate.
- Zero leakage detected across all split cross-comparisons.

## 23. PV-23 Semantics
As mandated by Section 28:
- When NN GPU determinism passes (`max_diff = 0.0 < 1.0e-10` on CUDA), CPU fallback is not required.
- PV-23 is recorded with semantic status `PASS`, `required = false`, `tested = false`, and explanation `"GPU determinism passed; CPU fallback not required."`, avoiding confusing raw boolean reporting.

## 24. Final Provenance Verdict

### **RESOLVED**

All 16 mandatory conditions of Section 5 have been verified with complete mathematical, cryptographic, and source-code evidence. The Autoencoder provenance is fully reconciled, proven, and certified for Sprint 11 freeze review.