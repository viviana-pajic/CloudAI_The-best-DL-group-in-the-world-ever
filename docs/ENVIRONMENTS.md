# Environments

Read saved notebook outputs first. Use the matching environment when rerunning.

| Work | Python | Requirements |
|---|---|---|
| Mushrooms 00-06 | 3.11.13 | `requirements-pycaret.lock.txt` |
| Citi Bike 00-10 | 3.12.10 | `requirements-citibike.lock.txt` |
| New notebooks and app | 3.12.10 | `requirements-review.txt` |

`scripts/setup_review.ps1` installs the new CPU app environment and registers its Jupyter kernel. PyTorch 2.11.0 is needed for saved GRU inference; inference works on CPU. Exact GRU training needs CUDA. See [AWS steps](AWS.md).

Do not run the PyCaret notebooks in the review environment. The recorded library versions differ. The small review requirements replace the older starter requirements for the current app.

Local model/API checks passed. The GitHub workflow installed the Linux CPU environment, rebuilt models, checked the app and built Docker successfully. Actual AWS installation and public Docker hosting remain unverified.
