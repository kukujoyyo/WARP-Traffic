# Pause both GPU queue workers after the job each one is currently running.
# Resume later with resume_queues.ps1.
$dir = $PSScriptRoot
foreach ($q in @("queue", "queue_b")) {
    $f = Join-Path $dir "$q.txt"
    $hold = Join-Path $dir "${q}_hold.txt"
    if (-not (Test-Path $hold)) {
        Copy-Item $f $hold
        Set-Content $f "# paused: pending jobs are in ${q}_hold.txt (restore with resume_queues.ps1)" -Encoding ascii
    }
}
Write-Host "paused: each worker finishes its current job, then starts nothing new"
