# Data and evaluation audit

This is an independent research audit outside the team repository. No course AWS resources were used. The original 1,000-row mushroom test and October-December Citi Bike test targets were excluded.

## Mushroom data

The current research split has 2,559 search rows, 640 threshold-calibration rows and 800 confirmation rows. One search row was quarantined after a feature-only duplicate check. This audit inspected confirmation labels only from saved result CSVs after root authorized and completed both one-time scorers. Earlier class work already used the 4,000 development rows, so this partition is a controlled development check rather than a previously unseen external test.

- Only 8 of the 3,199 allowed search/calibration rows have all 10 informative features present. Dropping rows with any null would remove almost everything.
- Search missingness: spore-print-color 95.35%, stem-surface 67.61%, cap-diameter 39.98%, stem-height 20.05%, stem-width 7.70%.
- Missing measurements were preserved. Search compared native numeric missing-value support, fold-fitted imputation and missing indicators. Absent categories used a missing token or mode filling as a measured ablation.
- Scale numeric values for logistic regression, SVM, KNN and neural networks. Fit scales inside each training fold. Scaling is usually unnecessary for tree splits.
- Search included dropping spore-print-color, guarded numeric ratios/logs and missing-count features. Controlled ExtraTrees results are recorded below.
- Numeric values contain no negative or infinite observations. Some stem measurements are zero. There is no source evidence that these zeros or large observations are incorrect; do not delete them solely because they look unusual.
- source_row_id is an identifier, not a predictor. The two jumbled_noise columns are only optional noise ablations.

Removing the noise columns revealed identical informative inputs. Search IDs 1770/3851 belong in the same CV fold. Calibration IDs 580/4082 have matching labels. Search 316 matched confirmation 3608; root excluded 316 without reading the confirmation label. The current blind collision audit finds no active search/calibration-to-confirmation duplicate.

There are no contradictory labels among identical complete informative inputs in the allowed rows. Many categorical-only patterns contain both labels, but those patterns omit numeric information. That does not establish mislabeled data or an irreducible error rate.

A controlled ExtraTrees experiment held model parameters constant while changing preprocessing one factor at a time. Eleven variants used the same five grouped folds under two seeds (110 fits). The iterative-imputation anchor and native missing-value support effectively tied: mean fold average precision 0.781427 versus 0.781398. Median filling, mode-filled categories, missing flags and removing the sparse spore field changed average precision by less than 0.2 percentage points. KNN filling reduced it by 1.08 points, removing logs/ratios by 1.71 points, and adding the noise columns by 1.75 points. This supports retaining informative missingness and engineered shape information without inventing a universal "best" imputation method. It is search-only evidence for this fixed ExtraTrees configuration, not a claim that the same effects hold for every model. Paired fold changes are recorded; overlapping folds are not independent repetitions.

## Poison detection

Accuracy alone is not a useful safety target. Treat poisonous as the positive class and report missed poisonous examples, poisonous recall, edible specificity and precision at the chosen threshold.

A lower threshold detects more poisonous examples but rejects more edible examples. Predicting "unsafe" for every sample gives 100% observed poisonous recall and 0% edible specificity; it does not identify mushrooms. Include that trivial baseline so a high-recall model has to demonstrate usefulness.

Choose thresholds using calibration predictions, then freeze them before confirmation scoring. Any threshold score measured on the same predictions used to choose it is a selection estimate. Do not repeatedly lower thresholds after viewing confirmation misses.

Zero misses in a finite sample cannot guarantee zero future misses. Frozen comparisons should report detected/total counts and nominal binomial intervals, while acknowledging prior development exposure and possible dependence between mushroom examples. Missing information and differences between the course data and real mushrooms remain limitations. This classifier is a class exercise and cannot decide whether a wild mushroom is safe to eat.

## Citi Bike data

The current problem predicts one next-day citywide count. Millions of raw trips become one row per day. January-June has 181 days; the 28-day history warmup leaves 153 training targets. The three tuning folds check 21 days each, so only 63 historical forecast dates directly guide the large hyperparameter search.

The January-June daily table has no null counts, missing dates, duplicated dates, negative/noninteger counts or zero-count days. Keep real low-demand and high-demand days. They may reflect weather, holidays or events and are the cases a forecasting model must handle.

All ride features must end at the previous day. The forecast assumes that yesterday's completed total is available before today's prediction. This is a rolling one-day task, not a three-month forecast made once in June. Actual previous validation-day totals are permissible under that contract.

Independent perturbation checks passed for all 153 training prediction dates: changing the target and every later count did not change features through that prediction date. July-September targets were not needed for this check. Median/mode/scale fit checks for the mushroom pipelines also passed. The TabM neural implementation independently passed exact shared 3/5-fold membership checks, group-safe inner early stopping, training-only preprocessing and mean-of-member-probabilities checks. The 800 confirmation feature rows contain no exact duplicates; their labels were not read by these checks.

An implementation review found a HistGradientBoosting preprocessing error: numeric missing indicators moved the ordinal categorical columns, while the old mask used their original positions. The model agent fixed the mask using the actual transformed width and preserved affected exploratory logs for reruns. The fix passed 18 native/median/mean/KNN/iterative checks with plain/engineered features and flags where applicable. JSON class-weight roundtrips and estimator cloning also passed. No confirmation predictions were needed.

The neural finalizer's source review passed: search-only selection and five-fold rechecks choose candidates; the candidate manifest is written before calibration is loaded; median inner-selected epochs control full-search training; only the separate 640 calibration rows set thresholds. A random-weight checkpoint roundtrip passed on CPU with CUDA hidden, both with and without numeric embeddings. Review exposed an unconditional Windows-only locking import in the inference dependency chain. Root fixed it with platform-specific Windows/POSIX locking. The POSIX branch was reviewed, but no AWS/Linux runtime was available for an execution check.

Both actual frozen neural bundles also passed separate fresh CPU processes with CUDA hidden. Their 640 saved calibration probability rows match within 3.0e-8; metadata matches the frozen manifest and 2,559/640 row counts; stored preprocessing exactly matches a search-only refit; checkpoint tensors are on CPU; and adding or changing source identifiers/noise leaves predictions unchanged. These checks never loaded class labels or computed comparative calibration metrics. Portability requires the inference/source dependency layout and pinned packages with the bundle, not only its weights file.

Before execution, the neural confirmation scorer passed source review after adding exact freeze/calibration checksums and an exclusive once-only marker. It records artifact hashes before opening the 800-row confirmation partition, reloads saved CPU bundles and applies the frozen thresholds directly. It contains no fitting or threshold-search calls. Root subsequently authorized and completed that single scoring run after accepting both finalist freezes. This audit reviewed its saved results and did not execute the scorer.

The classic finalizer also passed source review. Exact candidate configurations, hashes and finite mean/max ensemble plans are declared before calibration; fitting reuses those captured configurations on search only. CPU scoring and artifact-reload parity precede threshold selection. A successful freeze cannot be overwritten, and partial resumes require the same original declaration. All six actual classic bundles later passed fresh CPU reload checks: their 640 calibration scores match the saved float64 values exactly, and selected five-fold/repeated-seed lineage uses the current source IDs and group folds.

Citi Bike GRU/TCN code also passed six counts/weather and outer-fold structural checks. Sequences use only the previous 28 counts. The last 28 outer-training dates choose epochs, with training-only scalers and a fresh outer refit. TCN cropping passed a future-slot perturbation check. Selection routing preserves the expanded 518-row history and the separate forecast input contract; logs retain inner dates, epoch history, frozen epochs, model size and device. These checks did not run neural training or score July-September.

Hundreds of candidate searches can overfit a small collection of development dates. Freeze a short list based on chronological training CV before reviewing July-September. Keep October-December for the eventual agreed final evaluation. Report MAE in rides/day, per-fold stability, RMSE, WAPE and bias alongside model size and input requirements.

## Optional archived forecast inputs

Weather assistance is a separate candidate because the existing app accepts ride history and dates, not weather forecasts. Both archives below were fetched without an account, key or paid endpoint. Raw responses, URLs, timestamps, parameters and SHA-256 hashes are saved in audit/weather.

The GFS temperature archive provides mean/min/max temperature forecasts for 273 January-September dates. The primary columns use values predicted 48 hours before each valid hour. The optional 24-hour columns require a stronger publication-time assumption. The provider documents GFS 2 m temperature coverage from March 2021; most other model/variable archives begin in 2024. [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api)

JMA GSM precipitation and wind forecasts at 48-hour lead were also available for all 273 dates. GSM is a global model with about 55 km spatial resolution; its six-hourly fields are interpolated to hourly data. Do not describe these hourly outputs as independent hourly measurements. [JMA model documentation](https://open-meteo.com/en/docs/jma-api)

The JMA archive contained 12 precipitation forecasts of -0.1 mm. Negative rain amounts are physically invalid. Before any bike validation labels were inspected, root authorized flooring these small negative forecasts to zero. Raw hourly values, raw daily sums and negative-hour flags are retained. No dates were dropped. The cause of the archive anomaly is unverified.

Temperature and wind use instantaneous valid hours. Precipitation describes the preceding hour, so target-day totals use valid times from 01:00 through the following midnight and are grouped by interval-start date. Unix timestamps are converted to actual America/New_York dates; March 12 has 23 real hours. The API's fixed request-time UTC offset must not be applied across the whole year. [Timestamp and accumulation definitions](https://open-meteo.com/en/docs)

Lead offsets establish an initialization-time proxy, not the API's actual historical publication time. The 48-hour choice leaves a 24-hour buffer for the final precipitation interval on ordinary days and 23 hours on a 25-hour fall-DST day. Instantaneous temperature/wind retain at least 24 hours. These assumptions are documented and still need checking before a production input contract is claimed.

Weather features contain forecasts, not realized target-day weather or reanalysis. They were acquired without reading trip labels. Candidate selection still uses January-June folds; July-September remains a frozen comparison period. Data are CC BY 4.0: credit Open-Meteo and the originating model, link the license, and describe the daily aggregation/cleanup. [Open-Meteo license](https://open-meteo.com/en/licence)

## Optional extra training year

Root authorized one additional past year, 2022, as a separate research experiment. The exact official annual archive was found through the public bucket listing: 2022-citibike-tripdata.zip, 1,231,367,136 bytes. The completed audit accepted 365 dates and 29,838,442 NYC ride starts, including 276 December 2022 starts carried into January 2023 files. There were no missing dates or duplicated scoped ride IDs. October-December 2023 target members were never read in this process.

The accepted aggregate adds 365 past dates, leaving 518 training targets after the 28-day warmup, while keeping the original 2023 tuning and comparison windows. It may help seasonal patterns but also introduces a different year, fleet size and ridership level. Compare it separately, keep source provenance and preserve the existing 2023-only candidate. Do not create extra independent targets by randomly splitting individual trips from the same day. [Citi Bike system data](https://citibikenyc.com/system-data)

Independent current/future-count perturbations passed for all 518 added-history target dates. The original 2023 features match exactly, and the three check windows still run April 29-May 19, May 20-June 9 and June 10-30. Training rows increase to 455, 476 and 497 in those folds. The extended weather matrix has no missing or nonfinite values and uses the same targets and count features.

Matching archived forecasts were downloaded for 2022 using the same predetermined sources and rules. They cover every real NYC hour, including March 13's 23 hours and November 6's 25 hours. Six additional -0.1 mm forecast amounts on February 26 were floored by the existing rule and retained as raw values/flags. The extended feature files span 638 January 2022-September 2023 dates; no missing weather was invented for the added training year.

## Frozen Citi Bike comparison

Root authorized one July-September comparison after accepting the exact four-contract freeze. Independent checks reproduce all eight saved metric tables and 24 paired block-bootstrap calculations on the same 92 dates. Twelve exported bundles load and reproduce June probe predictions in a fresh process with CUDA hidden. No model was fitted or changed by this review, and October-December targets remain excluded.

The existing XGBoost control has MAE 14,761 rides/day and WAPE 12.24%. The 2023-only weather blend has MAE 13,494 (-8.58%); the expanded-history weather blend has MAE 13,263 (-10.15%). The two counts-only nominees have higher MAE than the existing control: 15,542 (+5.29%) and 14,834 (+0.49%). More search and a larger training history did not guarantee improvement with the original count inputs.

Every nominee's paired 95% descriptive interval includes zero with 3-, 7- and 14-day blocks. The expanded-history weather blend's seven-day interval for its MAE difference is [-3,648, +609] rides/day. Its lower point estimate is encouraging but does not establish a confident improvement, independent-test performance or future-season performance. Weather-assisted models also require the separate forecast input contract.

September is harder for all four nominees: their monthly MAE is about 20,927-23,156, compared with 9,267-12,532 in July/August. September 23's actual 36,069 rides were substantially overpredicted by every nominee. These are descriptive errors; no cause is established from a count drop alone, and no feature or threshold was changed after seeing them.

## Frozen mushroom confirmation

Root approved all six classic and two neural frozen candidates before their one-time scorers opened the same 800 development rows. This audit independently recomputed the saved results without loading a model, producing another prediction or changing a threshold. All 48 operating-policy rows and their exact Clopper-Pearson intervals match; the source, input, artifact and freeze hashes are unchanged. Opening markers follow the combined approval. The five-row search-only CLI example also has matching command, exit-code, input/output, model and source hashes.

The predeclared classic model is ExtraTrees v3_extra_trees_0094. At the ordinary 0.5 threshold it has 79.5% accuracy but detects only 181 of 303 poisonous rows. At its frozen threshold chosen for 100% calibration recall, it detects 300 of 303 (99.01%, descriptive 95% interval 97.13%-99.80%) and correctly leaves only 23 of 497 edible rows unflagged. It still misses three poisonous rows and flags 474 edible rows. The frozen high-recall neural candidate detects 291 of 303 at its corresponding threshold. These results expose the sensitivity/specificity tradeoff; they do not establish real-world edibility or justify changing the predeclared model after seeing confirmation.

The 800 rows have prior class-project exposure and are a controlled development comparison, not the untouched original 1,000-row final test. The audit did not access that reserved test. Frozen target recall describes the calibration policy, not a guarantee on later rows.

The final inventory refresh checks all 1,254 eligible broad and 480 cost-sensitive trial records, plus 32 five-fold rechecks and 30 additional-seed stability records. All 1,796 saved OOF source-ID tables use the current 2,559 search rows; 840 recorded fold fingerprints match reconstructed group folds. Superseded boundary and encoding cases are excluded. Older records retain their evidence gap: 956 broad screening records and the reference forest lack exact fold fingerprints. Their source IDs and fold row/class counts match, but this is not proof of recoverable individual fold assignments. The selected finalists have verified current five-fold and repeated-seed lineage.

## Files

- audit.json: partition-limited data statistics and fingerprints.
- blind_feature_collisions.json: current feature-only duplicate check.
- blind_feature_collisions_v2.json: initial boundary collision evidence.
- implementation_checks.json: causality and fitted-transformation verification.
- neural_implementation_checks.json: exact TabM folds, inner stopping, transforms and label-blind confirmation duplicate check.
- neural_finalizer_review.json: search/calibration ordering and platform-locking source review.
- neural_portability_checks.json: random-weight CPU checkpoint roundtrips with CUDA hidden.
- saved_neural_tabm_v3_0052_checks.json and saved_neural_tabm_v3_0028_checks.json: actual frozen-bundle CPU reload, lineage and probability-parity checks without labels.
- neural_confirmation_procedure_review.json: pre-execution static review of the checksum, once-only and frozen-threshold guards; this audit did not execute the scorer.
- classic_finalizer_review.json: exact pre-calibration configurations, freeze preservation, CPU threshold lineage and synthetic mixture checks.
- mushroom_inventory_lineage.json: actual current search IDs, recorded group-fold fingerprints, exclusion rules and five-fold/stability cache lineage.
- mushroom_inventory_metadata_gaps.json: older screening records whose exact fold assignment is not recoverable from their saved fields; historical records are preserved.
- classic_saved_artifact_preflight.json: actual six-bundle CPU calibration-probability parity, current fold/source IDs and exact core freeze checks.
- neural_authorization_preflight.json: exact 22-file sidecar and preserved neural artifact hashes, checked before confirmation approval.
- frozen_mushroom_results_review.json: authorized saved 800-row results, all 48 operating-policy metrics/intervals, unchanged lineage and CLI proof.
- bike_sequence_checks.json: GRU/TCN temporal scaling, causal convolution and selection-routing checks.
- bike_freeze_review.json: exact four-contract freeze, code/input/schema hashes, training-only blend parity and once-only summer scorer review.
- frozen_bike_results_review.json: authorized 92-date metrics, independent paired-bootstrap reproduction, residual summaries and actual CUDA-hidden CPU bundle checks.
- histgb_position_checks.json: corrected categorical positions, JSON class weights and estimator cloning.
- expanded_feature_checks.json: all 518 added-history causality checks and unchanged comparison dates.
- preprocessing_ablation.json/csv: exact controlled configurations and search-only results.
- preprocessing_ablation_matched_folds.csv: paired changes under identical folds.
- preprocessing_ablation.png: mean and individual matched-fold changes, without confidence-interval claims.
- frozen_comparison_plan.md: predefined paired block-bootstrap and residual review plan.
- uncertainty_checks.json: synthetic-only verification of paired error bootstrap behavior.
- weather/*: cached forecasts, feature tables and provenance.
- citibike2022/audit.json: completed added-year audit and source fingerprints.

The controlled preprocessing experiments and implementation checks are complete. Classic and neural mushroom finalists are frozen and their single authorized development comparison is independently checked. Four Citi Bike finalists are frozen and their single authorized development comparison is independently checked. No best-model claim is justified by the size of a search alone.
