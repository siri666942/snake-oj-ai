# run_judge.ps1

param(
    [int]$Times = 10,
    [string]$OutFile = ".\doc\judge_results.txt"
)

$dir = Split-Path $OutFile -Parent

if ($dir -and !(Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

"========== New Batch $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==========" | Tee-Object -FilePath $OutFile -Append

for ($i = 1; $i -le $Times; $i++) {
    "===== Run $i / $Times =====" | Tee-Object -FilePath $OutFile -Append
    python tools\judge.py --random | Tee-Object -FilePath $OutFile -Append
    "" | Tee-Object -FilePath $OutFile -Append
}