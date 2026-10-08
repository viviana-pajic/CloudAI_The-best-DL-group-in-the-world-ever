# Mushrooms: exploration and evaluation

**Owner: Viviana 
** Tasks 8 and 11 were implemented and executed on
8 October. The work extends the existing notebooks: task 8 is added to
`mushrooms/00_data_audit.ipynb` and task 11 to `mushrooms/02_baseline_models.ipynb`.
The code and initial explanations were prepared with AI assistance; the
[assistance record](AI_USE.md) describes that work.

## What these tasks achieve

**Task 8 decides what to clean.** The starter audit listed open questions:
noise columns, heavily missing columns, outliers and missing rows. Task 8
answers each with a statistical test or an experiment, so every cleaning
decision has evidence behind it.

**Task 11 decides how models are judged.** Before tuning, we fix the split, the
metrics and the decision threshold, so every model (ours, PyCaret's and the AWS
one) is compared the same way.

Both tasks use only the 4,000 development rows. The 1,000 reserved test rows
stay unseen to avoid snoop bias.

## Task 8: what the data shows

| Check | Result |
|---|---:|
| Development rows | 4,000 |
| Edible / poisonous | 2,485 / 1,515 (62% / 38%) |
| Rows without any missing value | 6 |
| Columns where missingness carries signal | 2 (`spore-print-color`, `stem-surface`) |
| Outliers by the IQR rule (cap, stem height, stem width) | 4.6% / 5.7% / 3.5% |
| Stems with height and width both 0 | 38 |
| Spearman correlation, cap diameter and stem width | 0.85 |
| Strongest category (Cramér's V) | `stem-surface`, 0.23 |
| Noise columns matching `cap-shape` row by row | 28% (pure chance: 28%) |

**Missing values:** almost every row has one, so deleting rows is impossible.
For most columns the gaps are random (chi-square p > 0.05). We impute inside the
model: the median for numbers and a `__MISSING__` category for text.

**Outliers:** sizes are right-skewed, and the largest caps form a group of
mostly edible mushrooms (22 of 25 above 40 cm), not typos. We keep them. Zeros
mean "no stem", not a missing value.

**Categories:** all codes are nominal (season is cyclic), so they are one-hot
encoded. Every real feature is related to the class, but weakly, so a model has
to combine many of them.

**Noise columns:** `jumbled_noise_0` and `jumbled_noise_1` have the same code
shares as `cap-shape`, but match it row by row only as often as chance. They are
shuffled copies with no information about the mushroom.

### The cleaning experiment

Two baseline models were trained on five versions of the data, on the same
cross-validation folds:

| Data version | Random forest ROC-AUC |
|---|---:|
| All columns | 0.821 |
| Without noise columns | 0.831 |
| Also without `spore-print-color` | 0.827 |
| `spore-print-color` as a "recorded" flag | 0.829 |
| Without noise columns and "was missing" flags | at least as good as version 2 |

Removing the noise columns gave the largest improvement, so they are dropped.
`spore-print-color` is kept, and the extra "was missing" columns for numbers are
dropped because they did not help.

**Separability:** out-of-fold probabilities overlap heavily between the classes,
and only 11 mushrooms are confident contradictions. There is no separate group
of obviously flipped labels; the data is simply noisy, so the threshold must be
chosen on purpose.

## Task 11: how models are evaluated

**Split:** the same 1,000 test rows stay locked away until the final models are
chosen. The other 4,000 rows use stratified 5-fold cross-validation, repeated
twice, with the same folds for every model (`cv_splitter()`).

**Metrics:** poisonous is the positive class. Models are ranked by average
precision (area under the precision-recall curve), with ROC-AUC as a second
score. Accuracy is not used, because "always edible" already scores 62%.

**Threshold:** every model uses the threshold that catches at least 90% of
poisonous mushrooms, chosen on out-of-fold predictions. At that threshold we
compare specificity: how many edible mushrooms are kept.

| Model | Average precision | Specificity at 90% sensitivity |
|---|---:|---:|
| Majority baseline | 0.379 | 0.00 |
| Logistic regression | 0.631 | 0.23 |
| Random forest | 0.783 | below 0.5 |

The random forest is clearly the best, but it overfits and still rejects more
than half of the edible mushrooms at 90% sensitivity. Tuning (tasks 20 and 21)
should raise average precision so that more edible mushrooms are kept.

## Run the work

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then open `mushrooms/00_data_audit.ipynb` and `mushrooms/02_baseline_models.ipynb`,
select the `.venv` kernel and click Run All. The cleaning experiment in
notebook 00 takes one to two minutes.

Useful explanations for the oral:

- "The noise columns are shuffled `cap-shape` values; removing them improved ROC-AUC."
- "Missing values are mostly random, so we impute instead of deleting rows."
- "We use cross-validation because 5,000 rows is too few to waste 1,000 on validation."
- "We rank by average precision because the poisonous class is what matters."
- "The threshold catches 90% of poisonous mushrooms; specificity shows the cost."

Next: task 13 applies these decisions in `mushrooms/01_data_preparation.ipynb`.