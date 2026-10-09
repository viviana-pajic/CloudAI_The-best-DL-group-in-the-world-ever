# Mushrooms: tuning logistic regression and random forest

**Owner: Viviana.** Tasks 20 and 21 were implemented and executed on
9 October 2026 with AI assistance. Task 20 is in
[`05_tune_logistic_regression.ipynb`](../mushrooms/05_tune_logistic_regression.ipynb)
and task 21 in [`06_tune_random_forest.ipynb`](../mushrooms/06_tune_random_forest.ipynb).
The [AI assistance record](AI_USE.md) describes that work.

Both tasks reuse Muneeb Shakoor's integrated workflow (`load_development`,
`candidates` and `evaluate_model` in `src/mushroom_workflow.py`) and the pinned
Python 3.11 environment. The tuned results therefore use the same 4,000
development rows, the same repeated folds (5 folds x 2 repeats, seed 42) and the
same metrics as the baselines in notebook 02. The reserved test rows are never used.

## Method: nested cross-validation

Tuning on the same folds that are reported would make the best setting look
better than it is. Each of the 10 outer folds therefore runs its own grid search
with 3 inner stratified folds, using only the outer training rows. The chosen
setting is then scored on the untouched outer validation fold. The result
estimates the whole procedure "search, then fit", not one lucky setting.

Average precision (AP) is the search and ranking metric, as fixed in task 11. A
final search on all development rows shows which settings the procedure prefers;
its inner scores are slightly lower than the outer scores because each inner
fold trains on fewer rows.

## Task 20: logistic regression

Grid: `C` in 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and `class_weight` None or balanced
(14 settings).

| Model | Mean fold AP | AP std | Mean fold ROC-AUC | Mean training AP |
|---|---:|---:|---:|---:|
| Baseline LR | 0.6307 | 0.0103 | 0.6894 | 0.6538 |
| Tuned LR (nested) | 0.6306 | 0.0105 | 0.6896 | 0.6538 |

The search prefers `C = 10` without class weights (inner AP 0.6254), but the top
six settings lie within 0.0013 of each other, less than their standard
deviations. `class_weight='balanced'` never helps: it shifts probabilities but
does not improve the ranking that AP measures.

Tuning does not improve logistic regression. The training score is barely above
the validation score, so the model underfits rather than overfits: a linear
model cannot combine the weak features that task 8 found, whatever its
regularisation. This negative result is kept as evidence that the limit is the
model family, not its settings.

## Task 21: random forest

Grid: `min_samples_leaf` in 1, 3, 5, 10 and `max_features` in sqrt, 0.3, 0.5
(12 settings). The number of trees stays at 150, as in the baseline. The
baseline forest has a large gap between training AP (0.951) and validation AP
(0.783), so we expected larger leaves to help.

| Model | Mean fold AP | AP std | Mean fold ROC-AUC | Mean training AP |
|---|---:|---:|---:|---:|
| Baseline RF | 0.7831 | 0.0087 | 0.8334 | 0.9507 |
| Tuned RF (nested) | 0.7885 | 0.0109 | 0.8412 | 1.0000 |

The search prefers `max_features = 0.3` with `min_samples_leaf = 3` (inner AP
0.7787), essentially tied with `min_samples_leaf = 1` (0.7781). Larger leaves
(5 and 10) rank lower.

Because both models use identical folds, they can be compared fold by fold:
tuning is ahead in **7 of 10 folds**, with a mean difference of +0.0054 AP. The
largest gains are +0.023 and +0.018; three folds are slightly worse (at most
-0.008).

Tuning improves the random forest slightly. The gain comes from
`max_features = 0.3` (about 19 of the 64 prepared columns per split instead of
about 8 with `sqrt`): after one-hot encoding most columns are sparse and weak,
so looking at more of them per split finds useful splits more often.

Larger leaves did not help, which contradicts our starting idea that limiting
overfitting would improve the forest. The outer searches mostly chose fully
grown trees, so the training score rose to 1.00 while validation still improved.
For a random forest, averaging many different deep trees generalises even when
each tree memorises its sample, so the validation score, not the training gap,
is what counts. The gain is smaller than the fold-to-fold spread, so we treat the
tuned forest as a modest improvement and the leading candidate for task 28.

## Limits

- The grids are small and hypothesis-led, not an exhaustive search; better
  settings may exist outside them.
- Nested CV estimates the tuning procedure on development data. The reserved
  test set is still unused and remains the only independent check (task 31).
- AP compares rankings. The operating threshold is chosen separately with the
  task-24 policy and must be re-checked for whichever model is finally chosen.

## Reproduce

In the Python 3.11 environment (`.venv-pycaret`, see
[MUSHROOM_WORKFLOW.md](MUSHROOM_WORKFLOW.md)), open notebooks 05 and 06, choose
the `.venv-pycaret` kernel and run all cells. Notebook 05 takes under a minute.
Notebook 06 trains about 360 forests and can take up to an hour on a laptop.
Results are saved to `reports/mushrooms/tuned_logistic_regression_*.csv` and
`reports/mushrooms/tuned_random_forest_*.csv`.

Useful oral explanations:

- "Nested CV keeps the tuning choice separate from the score we report."
- "Tuning did not help logistic regression because it underfits; the limit is linearity."
- "For the forest, more features per split helped; bigger leaves did not."
- "A training score of 1.00 is normal for deep forests; only validation counts."
- "The forest's gain is small, but it holds in 7 of 10 identical folds."