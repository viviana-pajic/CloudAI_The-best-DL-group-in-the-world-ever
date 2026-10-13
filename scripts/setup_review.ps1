$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Check-Step { if ($LASTEXITCODE -ne 0) { throw 'Setup failed. Keep the error shown above.' } }
py -m pip install --user uv
Check-Step
py -m uv python install 3.12.10
Check-Step
py -m uv venv .venv-review --python 3.12.10 --seed
Check-Step
py -m uv pip install --python .venv-review\Scripts\python.exe -r requirements-review.txt
Check-Step
py -m uv pip install --python .venv-review\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
Check-Step
.\.venv-review\Scripts\python.exe -m pip check
Check-Step
.\.venv-review\Scripts\python.exe -m ipykernel install --user --name cloudai-review --display-name 'CloudAI review (Python 3.12.10)'
Check-Step
Write-Output 'Ready. Run: .\.venv-review\Scripts\python.exe app/review_server.py'
