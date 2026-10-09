# AI assistance record

## Scope

AI assistance produced the initial Python helpers, notebook scaffolds and explanations, baseline experiments, local API and browser interface, README and workflow visuals. It also helped interpret course material and propose a task backlog. The starter was prepared on 4–5 October 2026.

The team selected this as a shared starting point. Creating the repository and discussing the project does not imply that all generated code has been reviewed. Individual reviewers, changes and reasoning will be recorded in the notebooks as work proceeds.

## Representative requests

These are shortened summaries of actual requests in the development conversation:

1. Analyse the class material and assignment, explain the project and prepare a local starter without pushing it.
2. Check the teacher's announcement requiring the replacement mushroom dataset.
3. Prepare meeting points and split the project into an ordered backlog with sprints.
4. Prepare a clean starter for the team repository, explain the files and guide the first push under Tomislav's GitHub account.

## Verification and limits

The replacement CSV fingerprint, mushroom notebook execution, development metrics, split separation and local inference were checked. The reserved mushroom final test has not been evaluated. PyCaret was a template in the original starter; the executed Mushroom continuation is recorded below. AWS, public hosting and automatic deployment remain future work.

## Citi Bike continuation, 6 October 2026

Tomislav requested: “Can you do tasks 7 and 9 and elaborate on them?” AI assistance implemented and executed code-driven acquisition, the global ride-ID audit, daily aggregation, two notebooks, exploratory plots and explanations. The actual data exposed ending-month file boundaries; January 2024 was added to recover December 2023 starts that finished after New Year. Source fingerprints, schema consistency, totals, calendar coverage and synthetic failure cases were checked.

Demand exploration uses January–September 2023, with October–December left aside as a candidate holdout pending team agreement. At this stage, no hypothesis test, forecast training or final-test evaluation was performed. The work is available locally for team review; this record does not imply that Tomislav or another member has already reviewed or independently written the implementation.

## Tasks 7 and 9 update preparation, 7 October 2026

Tomislav selected acquisition and exploration for this update and requested clear findings and an owner-based task checklist. AI assistance helped edit the explanations and prepare the local commit. Tomislav is responsible for tasks 7 and 9, and is assigned task 10 for a separate update.

The update contains two executed Citi Bike notebooks, acquisition code, source and audit snapshots, three exploration figures and the walkthrough. Task 10 files are kept outside this checkout until their separate update. Viviana's tasks are marked assigned, with her contribution pending, rather than completed. No forecast training or final-test evaluation is included.

Record substantive human reviews, changes and additional prompts here as development continues. Each member should be able to explain the submitted work and its limitations.

## Mushroom tasks 8 and 11, 8 October 2026

Viviana completed tasks 8 and 11 with AI assistance (Claude), which proposed the tests, experiments, evaluation rules and helper code in `src/mushrooms.py`; Viviana decided where the work belongs in the existing notebooks, removed redundant cells, chose the explanations, and ran and checked every result herself, without using the reserved test rows.

## Mushroom integration and tasks 13, 15, 18, 24, 8-9 October 2026

**Contribution owner: Muneeb Shakoor.** Muneeb requested the repository review,
authorized integration on `main`, required preservation of teammates' work and
historical evidence, and selected the final four-task scope. After an interruption,
he explicitly requested completion of preparation, reproduced baselines, actual
AutoML and meaningful development-only error/threshold analysis before stopping.

Codex generated/edited the implementation, installed the local Python environments,
ran the experiments and tests, investigated reduced-feature collisions, corrected
unsupported statistical wording, and prepared the notebooks/documentation.
The decisions and evidence are in [MUSHROOM_WORKFLOW.md](MUSHROOM_WORKFLOW.md).
This attribution identifies Muneeb's integration responsibility; it does not claim
he independently authored the generated code or has already completed a detailed
human code review. Those reviews should be recorded when they actually occur.

The work uses Viviana Pajic's tasks 8/11 as its EDA/evaluation foundation and the
shared AI-assisted starter maintained by Tomislav Novosel. The original baseline
notebook is retained as a historical snapshot, including Viviana's experiment
outputs. Contributor credit is retained in the current notebooks. Historical
results are distinguished from the common integrated protocol.

Actual work includes a frozen ten-feature preparation contract; repeated-CV dummy,
LR and RF reproduction; PyCaret screening of four complete pipelines; shared-fold
Extra Trees/boosting evaluation; development collision sensitivity; confusion
matrices and missingness groups; and a threshold procedure chosen on inner OOF
scores and evaluated on outer development folds. No model hyperparameter search
was run. The 90% recall target is an explicitly provisional team preference.

Saved notebook outputs, CSV/JSON reports, source/version records and nine tests
verify the implemented checkpoint. The earlier authorized collision investigation
checked label agreement within colliding groups, including reserved rows; no
reserved-test model predictions or performance evaluation were performed. The
latest completion did not extend the existing app integration, tune models,
start other project tasks, commit, merge or push.

## Citi Bike hypothesis continuation, 7 October 2026

Tomislav requested: “Do the same for Task 10, goal.” AI assistance formulated and implemented a within-week weekday/weekend contrast, a stationary-bootstrap uncertainty calculation, an executed notebook, figures, numerical snapshots and the explanatory guide. Because task 9 EDA had already examined these development data, the analysis is explicitly exploratory, rather than a prospective confirmatory test.

The run uses 38 complete weeks within January-September 2023, 20,000 resampled sequences, seed 42, and planned expected run lengths of two, four and six weeks. Verification checks known synthetic effects, date/count guards, the resampling mechanics, reproducibility, and invariance to changed October-December counts. The saved outputs and figures are checked. No forecast training or final-test evaluation occurs in this task, and no new push is made. Team review, scope acceptance and substantive human contributions remain to be recorded.

## Task 10 publication preparation, 9 October 2026

Tomislav requested that Codex pull the team's latest work and push the prepared continuation from his desktop repository. The held Task 10 files were restored after a fast-forward pull, without replacing teammates' mushroom work. AI assistance updated the README findings and checklist, normalized long dashes in the new text, and checked the cached daily-data fingerprint, numerical reproduction, hypothesis safeguards, notebook outputs and documentation links.

Task 10 provides exploratory evidence for a calendar pattern, not forecast accuracy. Task 12's split decision and the forecast models remain open. The contribution owner is Tomislav; substantive human review and any subsequent changes should be recorded when they occur.


## Mushroom tasks 20 and 21, 9 October 2026

Viviana completed tasks 20 and 21 with AI assistance (Claude), which proposed the nested cross-validation design, the parameter grids and the notebook code built on Muneeb's `mushroom_workflow.py`; Viviana set up the pinned Python 3.11 environment, ran both notebooks, added a fold-by-fold comparison to check whether the forest's gain is consistent, and checked every result herself, without using the reserved test rows.

## Citi Bike forecasting continuation, 9 October 2026

Tomislav requested completion of the remaining Citi Bike work and explicitly suggested using BESTIJA's GPU and GridSearch. AI assistance implemented chronological evaluation, features, actual FLAML screening, tuned Ridge/random forest/seasonal/XGBoost experiments, error analysis, saved candidate rebuilding, notebooks, app integration, a GitHub rebuild workflow and AWS/hosting preparation. XGBoost used the RTX 5080; the saved configuration confirms CUDA execution. The owner is Tomislav, and this does not imply independent human authorship or completed teammate review.

The implementation was checked for past-only features, training-fold preprocessing, saved-model/API agreement, invalid inputs, notebook execution, development-source fingerprints and CPU rebuilding without the raw archives. Both existing mushroom and new Citi Bike model paths were exercised through the local HTTP gateway. Browser automation was unavailable in the remote session. The mushroom experiment source and results were preserved; its existing provisional reference bundle was recreated in its separate Python 3.11 environment.

Model searches use January-June 2023 folds. July-September is development validation already seen during earlier EDA, and is used for local selection. The local candidate is refitted on eligible January-September dates for the demo, with its earlier validation scores clearly labelled. No final-test model predictions or scores were computed. The app example uses made-up counts for a January 2024 scenario.

The eight new local notebooks review the saved actual experiment runs by default and provide an explicit training mode. The AWS notebook is prepared without execution outputs; AWS access, the actual cloud run, public hosting, deployment wiring, frozen release choices and reserved-test evaluation remain unfinished. Record substantive human reviews, explanations and later changes here as they occur.
