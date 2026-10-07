# CloudAI

**The best DL group in the world ever**

Viviana Pajic · Tomislav Novosel · Muneeb Shakoor

**Deadline: 15 October** · **Oral: 23 October**

## Running the project

Use Python 3.12. Open PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/mushrooms.py
.\.venv\Scripts\python.exe app/server.py
```

Open **http://127.0.0.1:8765**. Stop it with **Ctrl+C**.

For notebooks, choose `.venv\Scripts\python.exe` as the kernel in VS Code and run them in order. Keep raw data, models and `.venv` out of Git; the scripts recreate them.

## What's in the folders?

| Folder | What's there |
|---|---|
| `mushrooms/` | Data checks, preparation, baseline models and an AutoML template |
| `citibike/` | Data download and exploration notebooks, source records and plots |
| `src/` | Python helpers used by the notebooks |
| `app/` | The local mushroom prediction page and server |
| `docs/` | More detailed explanations and the [AI assistance record](docs/AI_USE.md) |

## Mushrooms so far

We use the teacher's replacement dataset: **5,000 rows and 13 columns**. The source is recorded in [SOURCE_DATASET_VERSION.json](SOURCE_DATASET_VERSION.json).

These are validation results, using 3,000 training rows and 1,000 validation rows. The remaining 1,000 rows are kept for the final test.

| Model | Accuracy | Poison recall |
|---|---:|---:|
| Majority baseline | 62.1% | 0.0% |
| Logistic regression | 69.2% | 35.4% |
| Random forest | 75.1% | 42.0% |

The forest does best so far, but it still misses many poisonous examples. We need to look at the errors and tune the models. See [the baseline notebook](mushrooms/02_baseline_models.ipynb).

The AutoML notebook is still a template. Use a separate compatible environment, such as Python 3.11, following the lecturer's [PyCaret setup](https://github.com/mjochen/CloudAI/blob/master/Exercises/3%20model%20quality/5.1%20-%20Install%20PyCaret.ipynb).

## Citi Bike so far

**Tasks 7 and 9 - Tomislav.** We downloaded and checked the 2023 NYC data, then counted rides by their start date. We also needed January 2024 because it contains **410 rides that started in December 2023**.

The daily table has **35,107,120 rides across all 365 days of 2023**. We found no duplicate ride IDs or missing dates.

```powershell
.\.venv\Scripts\python.exe src/citibike.py
```

The first run downloads about 2 GB. Start with [notebook 00](citibike/00_data_acquisition.ipynb), then [notebook 01](citibike/01_data_audit.ipynb). Their outputs are already saved.

![Daily Citi Bike rides, January-September 2023](citibike/figures/daily_demand.png)

What we noticed:

- Average daily rides rise from about **58,000 in January** to **128,000 in August**.
- Wednesdays average about **107,000 rides**, compared with **84,000 on Sundays**.
- Recent counts and the count from a week earlier look useful to investigate.
- Some days have big drops, so weekday alone won't explain everything.

This makes the calendar and recent ride counts a reasonable starting point for predicting the next day's total. Seasons and people's routines might explain some of the patterns, but we haven't established why they happen or how well a model will predict them.

We explored January-September and left October-December aside for possible final testing. Task 12 will fix the evaluation split. More details are in [the tasks 7 and 9 notes](docs/CITIBIKE_TASKS_7_9.md).

## Tasks

### Planning

- [x] **01. Confirm requirements and dates.**
- [x] **02. Complete repository access and review the starter.**
- [ ] **03. Check everyone's environment** - local pilot works; team runs and AutoML setup still need checking.
- [x] **04. Agree scope and responsibilities.**
- [ ] **05. Check AWS and hosting access** - access, costs and cloud requirements.

### Sprint 1 - 6-8 October

- [x] **06. Check the replacement mushroom dataset.**
- [x] **07. Download and assemble Citi Bike data - Tomislav.**
- [ ] **08. Explore mushroom data - Viviana.**
- [x] **09. Explore and aggregate Citi Bike data - Tomislav.**
- [ ] **10. Test the Citi Bike hypothesis - Tomislav.**
- [ ] **11. Define mushroom evaluation - Viviana.**
- [ ] **12. Define Citi Bike evaluation.**
- [ ] **13. Finish mushroom preparation** - starter works; review cleaning choices.
- [ ] **14. Finish Citi Bike preparation.**
- [ ] **15. Reproduce mushroom baselines** - starter results are ready to review.
- [ ] **16. Build Citi Bike baselines** - compare with last week's count.
- [ ] **17. Connect both models to the app** - mushroom part works; Citi Bike still needed.

### Sprint 2 - 9-11 October

- [ ] **18. Run mushroom AutoML comparison.**
- [ ] **19. Run Citi Bike AutoML comparison.**
- [ ] **20. Tune mushroom model 1 - Viviana.**
- [ ] **21. Tune mushroom model 2 - Viviana.**
- [ ] **22. Tune Citi Bike model 1.**
- [ ] **23. Tune Citi Bike model 2.**
- [ ] **24. Investigate mushroom errors and choose a threshold.**
- [ ] **25. Investigate Citi Bike errors and improve features.**
- [ ] **26. Train and tune a mushroom model on AWS.**
- [ ] **27. Confirm or complete Citi Bike AWS training.**
- [ ] **28. Choose the final models.**
- [ ] **29. Host the app** - start once both prediction paths work.

### Sprint 3 - 12-14 October

- [ ] **30. Automate model updates and deployment.**
- [ ] **31. Evaluate the final models on the reserved test sets.**
- [ ] **32. Finish notebooks and documentation.**
- [ ] **33. Check the complete project from a fresh environment.**
- [ ] **34. Prepare the presentation and demo.**
- [ ] **35. Submit the repository link through Canvas.**

### Before the oral - 16-22 October

- [ ] **36. Review each other's notebooks.**
- [ ] **37. Practise the demo and questions.**
- [ ] **38. Check hosting and restart instructions for the assessment.**

Aim to finish on **14 October**, leaving a day for submission and fixes. Everyone should be able to explain both datasets.

Course assignment: [CloudAI project](https://github.com/mjochen/CloudAI/blob/master/Discussion%20topics/project%20assignment.md).

The starter and these notebooks were prepared with AI assistance; details are in [docs/AI_USE.md](docs/AI_USE.md). The mushroom dataset is coursework data; its predictions cannot establish whether a real mushroom is edible.
