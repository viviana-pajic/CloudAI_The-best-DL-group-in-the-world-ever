# Mushrooms: exploration and evaluation foundation

**Owner: Viviana Pajic.** Tasks 8 and 11 were implemented and executed on
8 October 2026 with AI assistance. Task 8 is in
[`00_data_audit.ipynb`](../mushrooms/00_data_audit.ipynb); the original task-11
code and outputs are preserved in the
[historical baseline notebook](../mushrooms/history/02_baseline_models_before_integration.ipynb).
The [AI assistance record](AI_USE.md) preserves Viviana's contribution and review.

Muneeb Shakoor's later integration builds on these decisions, with Codex assistance.
This note preserves the historical evidence while correcting interpretations and
distinguishing it from the [current workflow](MUSHROOM_WORKFLOW.md). No original
EDA experiment has been rerun or erased for this documentation update.

## Task 8: what the development data showed

| Check | Historical result |
|---|---:|
| Development rows | 4,000 |
| Edible / poisonous | 2,485 / 1,515 |
| Rows without any missing value | 6 |
| Columns with detected missingness/class association | 2 (`spore-print-color`, `stem-surface`) |
| Outliers by the IQR rule (cap, stem height, stem width) | 4.6% / 5.7% / 3.5% |
| Stems with height and width both 0 | 38 |
| Spearman correlation, cap diameter and stem width | 0.85 |
| Strongest reported categorical association (Cramer's V) | `stem-surface`, 0.23 |
| Noise columns matching `cap-shape` row by row | About 28%, similar to chance |

**Missing values:** almost every row has a missing entry, so complete-case deletion
would discard almost all development data. Non-significant missingness/class
tests do not prove MCAR or explain the missingness mechanism. Median numeric
imputation and categorical `__MISSING__` encoding are learned inside training folds.

**Unusual values:** sizes are right-skewed; 22 of the 25 caps above 40 in the supplied
measurement column were edible-labelled. This does not prove measurement validity.
Likewise, zeros cannot be confidently interpreted as "no stem" without source
documentation. The current policy retains these observations rather than asserting
their biological meaning or deleting them automatically.

**Categories and noise:** categorical codes are treated as nominal. Sparse expected
cell counts limit asymptotic chi-square interpretations. The two noise columns have
category proportions similar to `cap-shape` and chance-like row agreement; this is
consistent with shuffled/noise-like columns but does not establish their exact
generation process. Their removal is supported by the development ablation below.

### Historical cleaning experiment

Two baseline models were compared across five preparation variants on matched
development folds. Preserve these rounded RF ROC-AUC results as historical evidence:

| Data version | RF ROC-AUC |
|---|---:|
| All columns | 0.821 |
| Without noise columns | 0.831 |
| Also without `spore-print-color` | 0.827 |
| `spore-print-color` reduced to a recorded flag | 0.829 |
| Without noise columns and numeric missing indicators | At least as good as version 2 in that experiment |

These results motivate dropping the noise columns, retaining `spore-print-color`
and omitting numeric missing indicators in task 13. They do not prove a universal
optimal preparation. Overlapping OOF scores and confident model/label
contradictions do not establish label corruption; no labels are rewritten.

## Task 11: historical evaluation and later clarification

Viviana fixed the existing 1,000-row final-test membership, combined the other 4,000
rows for development, and used stratified five-fold CV repeated twice with shared
folds. Poisonous is the positive class. Average precision is the primary ranking
metric; ROC-AUC is secondary. AP is a recall-weighted precision summary, not
trapezoidal PR-AUC. Accuracy alone hides the majority baseline's zero poison recall.

The historical table reported AP approximately 0.379 / 0.631 / 0.783 for dummy /
LR / RF. Its threshold helper chose maximum precision subject to at least 90%
recall on development OOF predictions. Specificity was then reported using those
same predictions (roughly 0 / 0.23 / below 0.5). This is a **selection estimate**:
the target was not independently validated, and maximizing precision is not
identical to maximizing specificity.

The integrated task-15 comparison reproduces AP/ROC-AUC with a fixed 0.5 reference
threshold for error metrics. Task 24 separately evaluates a maximum-specificity
recall-constrained policy, choosing thresholds inside outer training folds and
evaluating on held-out outer development folds. The measured policy recall is
89.24%, demonstrating that a 90% calibration constraint is not a guarantee.
No tuning improvement is presumed; tasks 20/21 remain unstarted.

## Reading and reproducing the work

Use the historical notebooks to inspect the original evidence. For the current
reproducible implementation, follow [MUSHROOM_WORKFLOW.md](MUSHROOM_WORKFLOW.md)
and run notebooks 01-04 in the pinned Python 3.11 environment. Do not compare the
old 3,000/1,000 validation result directly with the integrated 4,000-row repeated-CV
result as though only the model changed.

Useful oral explanations:

- Removing the two noise columns helped in a matched development ablation; their exact origin is not proven.
- Imputation preserves rows without claiming that missingness is random.
- Shared folds and fold-fitted preprocessing make candidate comparisons consistent.
- AP measures ranking quality; a threshold controls the precision/recall trade-off.
- Development threshold targets and reserved-test performance are different claims.
