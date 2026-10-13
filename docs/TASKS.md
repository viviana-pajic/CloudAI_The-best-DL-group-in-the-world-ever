# Tasks

All six handoff commits are complete. Local checks and GitHub builds passed. Both CPU AWS notebooks ran on 13 October. The public app and an actual automatic update passed checks. Release selection and final tests remain pending.

## Completed work

| Owner | Work | Original task numbers |
|---|---|---|
| Team | Requirements, repository access, scope and replacement dataset | 1, 2, 4, 6 |
| Viviana | Mushroom EDA, evaluation and two tuned models | 8, 11, 20, 21 |
| Muneeb | Mushroom preparation, baselines, AutoML and errors | 13, 15, 18, 24 |
| Tomislav | Citi Bike data, hypothesis, preparation, models, errors and local app | 7, 9, 10, 12, 14, 16, 17, 19, 22, 23, 25 |

Task 3 is checked locally; fresh teammate reproduction belongs to task 33. Contributions and AI assistance are recorded in [AI use](AI_USE.md).

## Team review

- Everyone explains the results and the inputs their model needs.
- Review both completed AWS notebooks. Tomislav used the supplied CPU notebook in `us-east-1`; no personal AWS account or GPU upgrade was used.
- Review the public app and agree the release models before opening final tests.

## Remaining work

| Tasks | Work | Owner/status |
|---|---|---|
| 5 | Confirm AWS and hosting access | Done: university sandbox and free Render service |
| 26 | Run mushroom AWS notebook 10 and save evidence | Done: Tomislav, with Codex assistance; Muneeb review pending |
| 27 | Run bike AWS CPU notebook 16; GPU 17 optional | Done: Tomislav, with Codex assistance; GPU run optional |
| 28 | Choose and freeze release models and threshold | Team; pending |
| 29 | Host both prediction paths | Done: Tomislav; public app and all prediction options checked |
| 30 | Verify a GitHub-triggered model update | Done: Tomislav; passing workflow triggered Render and the new commit passed public checks |
| 31 | Score the reserved final tests once | Team; not run |
| 32-33 | Save final outputs and reproduce the hosted release | Team; pending |
| 34, 36-38 | Cross-review, prepare the demo, rehearse and check assessment hosting | Team; pending |
| 35 | Submit the repository link in Canvas | Team; pending |

Use the [release steps](RELEASE.md). Target: 14 October. Deadline: 15 October 2026, 23:59. Oral: 23 October.
