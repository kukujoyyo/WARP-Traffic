# Resume both GPU queue workers (A: queue.txt, B: queue_b.txt) where they stopped.
# Finished jobs are listed in *_done.txt and are skipped; nothing restarts from scratch.
# Usage (PowerShell, with the project environment activated):  & .\experiments\resume_queues.ps1
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$py  = (Get-Command python).Source                               # Python of the active environment
$log = Join-Path (Split-Path (Split-Path $PSScriptRoot)) "logs"  # <project>\logs, next to the repository

foreach ($q in @("queue", "queue_b")) {
    $hold = Join-Path $dir "${q}_hold.txt"
    if (Test-Path $hold) {                       # put the held job list back
        Copy-Item $hold (Join-Path $dir "$q.txt") -Force
        Remove-Item $hold
        Write-Host "restored $q.txt"
    }
    $pause = Join-Path $dir "$q.pause"
    if (Test-Path $pause) { Remove-Item $pause; Write-Host "removed $q.pause" }
}

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*queue.py*" }
$runA = $procs | Where-Object { $_.CommandLine -notlike "*queue_b.txt*" }
$runB = $procs | Where-Object { $_.CommandLine -like "*queue_b.txt*" }

if ($runA) { Write-Host "worker A already running (it picks up the restored queue by itself)" }
else {
    Start-Process cmd -ArgumentList "/c", "`"`"$py`" queue.py >> `"$log\queue_runner.log`" 2>&1`"" -WorkingDirectory $dir -WindowStyle Hidden
    Write-Host "started worker A"
}
if ($runB) { Write-Host "worker B already running" }
else {
    Start-Process cmd -ArgumentList "/c", "`"`"$py`" queue.py queue_b.txt >> `"$log\queue_runner_b.log`" 2>&1`"" -WorkingDirectory $dir -WindowStyle Hidden
    Write-Host "started worker B"
}
