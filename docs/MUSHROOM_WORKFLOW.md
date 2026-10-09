# Mushroom tasks 13, 15, 18 and 24

Completed development checkpoint, 9 October 2026. **Contribution owner: Muneeb
Shakoor**, with Codex assistance for implementation, execution and documentation.
This integrates **Viviana Pajic's exploration/evaluation (tasks 8/11)** and the
shared AI-assisted starter maintained by **Tomislav Novosel**. It does not claim
Muneeb independently authored their earlier work or has completed human code
review. See [AI_USE.md](AI_USE.md).

## Evidence of completion

| Task | Completed work | Reviewable evidence |
|---|---|---|
| 13 | Graph-free preparation, fixed membership, ten inputs, fold-fitted transformations, collision policy | [Notebook 01](../mushrooms/01_data_preparation.ipynb), [split manifest](../mushrooms/split_manifest.json), [contract](../reports/mushrooms/contract.json), [collision investigation](../reports/mushrooms/collisions.json) |
| 15 | Dummy, LR and RF reproduced using the same preparation, folds, class mapping and metrics | [Notebook 02](../mushrooms/02_baseline_models.ipynb), [baseline results](../reports/mushrooms/baseline_cv.csv), per-fold CSV and source/version metadata |
| 18 | Actual bounded PyCaret comparison, followed by common-fold evaluation of Extra Trees and histogram boosting | [Notebook 03](../mushrooms/03_automl.ipynb), [screening results](../reports/mushrooms/automl_screening.csv), [common comparison](../reports/mushrooms/comparison.csv) |
| 24 | Confusion matrices, poisonous-as-edible examples, missingness groups, threshold trade-offs, selected policy and separate development evaluation | [Notebook 04](../mushrooms/04_model_comparison.ipynb), [threshold experiment](../reports/mushrooms/threshold_run.json), error and threshold CSVs |

The checkpoint does not complete the whole course project. Tasks 20/21, final model
selection, reserved-test evaluation, AWS, Citi Bike and deployment remain outside
this work. Task 24 is complete as a **development policy investigation**; it is not
validation of a real-world safety policy.

## Reproduce

All current Mushroom experiments use **Python 3.11.13**, PyCaret 3.3.2,
scikit-learn 1.4.2, pandas 2.1.4, NumPy 1.26.4 and SciPy 1.11.4.
`requirements-pycaret.txt` lists direct dependencies;
`requirements-pycaret.lock.txt` freezes all 110 installed distributions. Keep this
separate from the existing Python 3.12 `.venv`, whose newer sklearn/Pandas versions
are not the recorded Mushroom experiment environment.

In this checkout the environment already exists. From the repository root:

```powershell
.\.venv-pycaret\Scripts\python.exe scripts/run_mushroom_notebooks.py
.\.venv-pycaret\Scripts\python.exe -m unittest discover -s tests -v
```

The notebook runner forces the current Python interpreter as its kernel, writes
executed outputs only after successful execution, and stops on cell errors. It
runs 01, 02, 03, then 04. Historical EDA/results are not rerun. To execute only a
changed notebook after its dependencies have run:

```powershell
.\.venv-pycaret\Scripts\python.exe scripts/run_mushroom_notebooks.py --only 04_model_comparison.ipynb
```

For a fresh checkout with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv-pycaret
.\.venv-pycaret\Scripts\python.exe -m pip install -r requirements-pycaret.lock.txt
```

The first notebook downloads the pinned teacher CSV if absent and verifies its
SHA-256. Data, OOF row predictions, model bundles and runtime caches are ignored by
Git. Small CSV/JSON experiment reports and executed notebook outputs are retained.
An interactive notebook must use `.venv-pycaret\Scripts\python.exe` as its kernel.
The local portable Python installation is under `.tools/python/`; it is not
registered with the system `py` launcher. It is not needed to rerun the existing
environment.

Equivalent script entry points, in dependency order, are:

```powershell
.\.venv-pycaret\Scripts\python.exe src/mushroom_workflow.py baseline
.\.venv-pycaret\Scripts\python.exe src/mushroom_automl.py
.\.venv-pycaret\Scripts\python.exe src/mushroom_error_analysis.py
```

Notebook 04 also generates the reference-model error tables and plots. The full
notebook runner is the preferred way to recreate all evidence.

## Preparation and evaluation contract

Teacher CSV SHA-256:
`1f1a25f2f330458ed95ce9f6ffe3241582312a42cb21797a2b7bd5e5ad281114`.
The original seed-42 stratified split remains exactly 3,000 train / 1,000 validation
/ 1,000 final test. Development concatenates the original sorted training IDs and
then the original sorted validation IDs, giving 4,000 rows: 2,485 edible, 1,515
poisonous. Source-row IDs are identifiers, never model inputs. `e=0`, `p=1`.

Ten inputs remain after dropping `jumbled_noise_0` and `jumbled_noise_1`:
`cap-diameter`, `stem-height`, `stem-width`, `spore-print-color`, `gill-color`,
`habitat`, `season`, `ring-type`, `cap-shape`, `stem-surface`. Keep all rows and
observed values, including rare categories, large measurements and zeros. The
dataset cannot establish whether unusual values are valid biological measurements.

Each fitted pipeline learns numeric medians and categorical imputation/one-hot
encoding using only its training fold. Text missingness becomes `__MISSING__`;
unseen categories are ignored by the encoder. LR additionally scales numeric
features. Numeric missing indicators are omitted following Viviana's development
ablation. There is no globally fitted feature matrix. Notebook 01 writes raw
model-ready inputs to ignored `data/processed/mushrooms/development.csv` and
illustrates transformation using only a training fold, without exploratory graphs.

The common model comparison uses **RepeatedStratifiedKFold, 5 folds x 2 repeats,
seed 42**. Fold fingerprint:
`2f607146ef2d173af2bd97bb34d3f74954eaaae3a4143d7023e6fec28a71a0b5`.
Rank by mean fold **average precision (AP)**; ROC-AUC is secondary. AP uses recall
increments to weight precision, not trapezoidal PR integration. Fold standard
deviations describe variation, not confidence intervals. Fixed-0.5 error tables
average two held-out probabilities per row; their pooled AP is a different
aggregation from mean fold AP. Never mix those definitions.

## Actual experiments and results

| Model | Mean fold AP | AP std (population) | Mean fold ROC-AUC | Mean training AP | OOF recall at 0.5 | OOF precision at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| RF, 150 trees, minimum leaf 3 | 0.783075 | 0.008699 | 0.833420 | 0.950684 | 0.520792 | 0.853896 |
| Histogram boosting, 100 iterations, 15 leaves, L2=1 | 0.758460 | 0.010047 | 0.812080 | 0.879463 | 0.536634 | 0.815446 |
| Extra Trees, 150 trees, minimum leaf 3 | 0.754268 | 0.012941 | 0.804034 | 0.921481 | 0.482508 | 0.844111 |
| LR, default regularization, 3,000 iteration cap | 0.630651 | 0.010340 | 0.689445 | 0.653796 | 0.345875 | 0.728790 |
| Majority dummy | 0.378750 | 0.000000 | 0.500000 | 0.378750 | 0 | 0 |

RF remains the leader among these settings. Its training/development gap is
substantial. Additional families did not beat it; keep these negative results as
model-selection evidence. Nothing here establishes the best possible settings of
any family. Fit times in the CSVs are machine-specific measurements, not controlled
hardware benchmarks. No model hyperparameter search was performed.

### AutoML limitations

PyCaret `ClassificationExperiment.compare_models` executed four complete sklearn
pipelines, ranked by an explicitly added AP metric. PyCaret's native preprocessing
is disabled; the shared pipeline fits imputation/encoding inside every CV fit.
The bounded set tests a linear reference, bagged trees, randomized trees and
boosting. It is actual AutoML screening, not a blank template or a manual leaderboard
labelled AutoML, but it is not an exhaustive library-wide search.

PyCaret screens on the original **3,000 training rows**, using repeated 5-fold CV.
Its explicit administrative `test_data` contains only the original **1,000
development validation rows**, not the reserved final test. No holdout prediction
is requested. PyCaret may downcast numeric dtypes. Its scores are therefore kept
separate from the common 4,000-row sklearn comparison.

| PyCaret screening model | AP | ROC-AUC |
|---|---:|---:|
| Random forest | 0.763011 | 0.823890 |
| Histogram boosting | 0.749369 | 0.813593 |
| Extra Trees | 0.739524 | 0.795718 |
| Logistic regression | 0.624164 | 0.693998 |

Extra Trees and boosting were then fitted on the identical 4,000-row folds used by
the reproduced baselines. Existing baseline results were reused only after checking
the complete contract/environment. Baseline and candidate reports record model
parameters and source hashes.

### Reduced-feature collisions

Raw predictors have no exact duplicates. Removing noise leaves five extra rows
across four groups: `{232,3164,4244}`, `{316,3608}`, `{580,4082}`, `{1770,3851}`.
Within each group, retained values, missingness and labels agree. Identical retained
predictors do not prove identical source observations; no source observation ID is
available. Keep all rows and unchanged test membership.

The development sensitivity analysis excludes only training rows whose retained
predictors match a validation row, keeping validation folds fixed. RF AP becomes
**0.784446** versus **0.783075**: delta **+0.001371**. ROC-AUC becomes **0.834341**
versus **0.833420**. Ten training-row exclusions occur across ten fits, not ten
distinct dataset rows. This small change does not establish source independence.

The earlier authorized collision-integrity check inspected agreement of labels in
colliding groups, including reserved rows. This is documented in `collisions.json`.
It was not model evaluation. No reserved-test probabilities, predictions, confusion
matrices or performance metrics have been computed.

## Error analysis and threshold policy

At 0.5, using averaged repeated OOF scores:

| Model | True edible -> edible (TN) | Edible -> poisonous (FP) | Poisonous -> edible (FN) | True poisonous -> poisonous (TP) |
|---|---:|---:|---:|---:|
| Random forest | 2,350 | 135 | 726 | 789 |
| Histogram boosting | 2,301 | 184 | 702 | 813 |

RF misses **47.92% of poisonous labels**, despite 85.39% precision among predicted
poisonous rows. These are different denominators. Boosting has higher recall at
0.5 but lower AP, so a threshold comparison cannot replace a ranking comparison.

| RF missing-feature group | All rows | Poisonous | Edible | FN | FP | FN / poisonous |
|---|---:|---:|---:|---:|---:|---:|
| 0-2 | 1,279 | 501 | 778 | 173 | 42 | 34.53% |
| 3-4 | 2,243 | 853 | 1,390 | 444 | 88 | 52.05% |
| 5-10 | 478 | 161 | 317 | 109 | 5 | 67.70% |

Missingness is associated with missed poisonous labels; this does not establish
causality or justify deleting rows. `error_examples.csv` gives source IDs, features
and scores for the most confident false negatives/positives of RF and boosting.
`threshold_false_negatives.csv` gives ten remaining poisonous-as-edible examples
under the separately evaluated policy. Model disagreement cannot prove label errors.

### Selected rule and actual held-out development result

Retain task 11's **90% poison-recall target** as a provisional coursework preference:
missed poisonous labels matter, and a lower threshold can trade more false alarms
for fewer misses. There is no supplied loss function, deployment constraint or
lecturer requirement establishing 90% as sufficient. Alternatives at 80% and 95%
are shown descriptively. We do not choose a separate threshold per missingness group.

Choose maximum specificity subject to calibration recall >= 0.90; break ties by
the largest observed threshold. Predict poisonous for score >= threshold. This
explicitly replaces the historical helper's maximum-precision objective.

To evaluate that procedure, use five stratified outer development folds. Within
each outer training set, obtain three-fold OOF RF scores and select a threshold;
refit the same RF on the outer training set, then score its untouched outer fold.
This is **20 fits with fixed model parameters**, not tasks 20/21 tuning. All splits
shuffle with seed 42, preprocessing stays inside fits, and each row gets one outer
held-out score. Audit records preserve outer row IDs and inner-fold fingerprints.

| Evaluation on the same outer scores | Poison recall | Poison precision | Specificity | FN | FP |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.5 | 52.54% | 84.32% | 94.04% | 719 | 148 |
| Threshold chosen within each outer training fold | **89.24%** | **49.82%** | **45.19%** | **163** | **1,362** |
| Single threshold selected on all OOF labels (apparent selection result) | 90.03% | 49.14% | 43.18% | 151 | 1,412 |

The selected single development threshold is **0.24692594**. Its row above
uses the same labels that selected it, so that row is not an independent estimate.
The outer-fold procedure evaluates thresholds between **0.245484 and 0.256004**,
chosen without their evaluation labels. It misses 163 poisonous rows and falsely
flags 1,362 edible rows. Recall ranges from 88.12% to 90.43% across outer folds;
only one of five folds reaches 90%. The chosen calibration constraint is plainly
not a held-out recall guarantee.

The outer policy reduces misses from 719 to 163 on the same scores, at the cost of
1,214 additional false alarms. Under this policy, false-negative rates by missingness
group are 9.58%, 10.79% and 14.29%; all denominators are saved in
`threshold_error_slices.csv`. These residual failures limit any claim of safety.

RF/preparation were already chosen using development experiments, so the outer
evaluation is conditional on those choices. It is not fully nested evaluation of
all model selection. Inner models train on fewer rows; their score distributions
and thresholds may shift when fitting on all development data. Scores are not
calibrated real-world risks. The reserved final test remains unevaluated.

The existing demo bundle remains at threshold **0.5**. The selected offline policy
is stored separately and has not been deployed. The earlier `contract.json` and
baseline metadata describe the preceding fixed-0.5 screening checkpoint; the
later `threshold_run.json` is the authoritative task-24 policy record.

## History, changed files and attribution

The original 3,000/1,000 baseline and Viviana's task-11 work are preserved in
`mushrooms/history/02_baseline_models_before_integration.ipynb`; the original
`reports/mushroom_validation_metrics.csv` is unchanged. Historical results used
different features/protocols and must not be presented as directly comparable
improvements. The old helper remains accessible via `src/mushrooms.py --historical`;
normal runs dispatch to the integrated workflow.

Files in this checkpoint:

- `src/mushrooms.py`: compatible shared feature/preparation helpers, frozen-split checks and current entry point; earlier helpers retained.
- `src/mushroom_workflow.py`: current preparation, common CV, collision sensitivity and provisional artifact generation.
- `src/mushroom_automl.py`: actual PyCaret screening and common-fold challenger evaluation.
- `src/mushroom_error_analysis.py`: development threshold selection/evaluation and error reports.
- `mushrooms/01_data_preparation.ipynb`, `02_baseline_models.ipynb`, `03_automl.ipynb`, `04_model_comparison.ipynb`: executed current evidence. `03_automl_template.ipynb` is replaced by the executed notebook.
- `mushrooms/00_data_audit.ipynb`: interpretation/credit corrections only; prior experimental code and outputs preserved.
- `mushrooms/history/02_baseline_models_before_integration.ipynb`, `mushrooms/split_manifest.json`: historical evidence and fixed row membership.
- `reports/mushrooms/*.csv` and `*.json`: compact actual experiment evidence, settings, folds and source/version provenance.
- `requirements-pycaret.txt`, `requirements-pycaret.lock.txt`: isolated, reproducible dependencies.
- `scripts/run_mushroom_notebooks.py`, `tests/test_mushroom_workflow.py`, `tests/test_mushroom_thresholds.py`: notebook execution, data isolation, preprocessing, threshold and provenance checks.
- `README.md`, `docs/MUSHROOM_WORKFLOW.md`, `docs/MUSHROOM_TASKS_8_11.md`, `docs/AI_USE.md`: accurate status, history, reproduction and contribution records.
- `.gitignore`: local environments/caches and generated data/models excluded; compact Mushroom reports retained.
- `app/server.py`, `app/index.html`: integration edits already present before the interruption, preserved. No new app features were added during this four-task completion.

## Verification and tomorrow's demonstration

All four current notebooks have executed outputs without cell errors. The final
resume reran notebook 01 and completed notebook 04; notebooks 02/03 already had
successful outputs matching unchanged implementation hashes, so their completed
model experiments were preserved. The original EDA and historical notebook are
historical evidence, not freshly rerun experiments.

Nine tests cover frozen membership, forbidden test IDs, fold-fitted imputation,
unknown categories, existing artifact/API parity, tied threshold scores, invalid
inputs, class denominators, inner/outer isolation and stored source provenance.
No reserved rows enter threshold calibration, model fitting or prediction.

The new threshold experiment was executed twice: maximum score difference
**2.22e-16**, identical fold thresholds and classifications. Stored common-fold
AP/ROC-AUC and all fixed-0.5 OOF metrics were independently recomputed from their
actual CSV artifacts. All 110 installed distributions match the lock file.
[verification.json](../reports/mushrooms/verification.json) records notebook hashes,
executed cell counts, source/metric consistency and this reproducibility check.

For the demonstration, show notebook 01's preparation contract, notebook 02's
reproduced baselines, notebook 03's actual PyCaret and common-fold tables, then
notebook 04's confusion matrices, missingness groups and threshold-policy results.

Suggested wording: **"I completed the Mushroom integration, reproduction, bounded
AutoML comparison and development error/threshold analysis, with AI assistance.
It builds on Viviana's EDA/evaluation and our shared starter with Tomislav. RF
reproduces AP 0.7831 and ROC-AUC 0.8334. Our recall-focused threshold policy reaches
89.24% recall on separate development folds but still misses 163 poisonous rows.
We have not evaluated the final test or performed model hyperparameter tuning."**

Three oral-defense concepts:

1. **Leakage and evaluation separation:** transformations learn only from training
   folds; threshold calibration has its own inner folds; the final test stays
   reserved. Earlier development model selection still limits independence.
2. **Ranking versus decisions:** AP/ROC-AUC compare score rankings; threshold
   changes alter precision, recall and the confusion matrix. Precision and recall
   have different denominators. Averaged OOF and mean fold scores are different.
3. **Reproducibility and uncertainty:** frozen data/splits/seeds/library versions
   make results reproducible, but do not remove overfitting, selection bias,
   distribution shift or false negatives. A 90% calibration target is not safety.
