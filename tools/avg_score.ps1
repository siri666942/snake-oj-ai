# avg_score.ps1

param(
    [string]$File = ".\doc\judge_results.txt",
    [int]$Start = 1,
    [int]$End = 0
)

if (!(Test-Path $File)) {
    Write-Host "文件不存在: $File"
    exit
}

# 提取所有 weighted_total
$scores = Select-String -Path $File -Pattern "weighted_total=([0-9.]+)" |
    ForEach-Object {
        [double]$_.Matches[0].Groups[1].Value
    }

$count = $scores.Count

if ($count -eq 0) {
    Write-Host "没有找到 weighted_total=xxx"
    exit
}

# 默认 End=0 表示到最后一个
if ($End -eq 0) {
    $End = $count
}

# 参数合法性检查
if ($Start -lt 1 -or $End -lt $Start -or $End -gt $count) {
    Write-Host "范围错误：当前共有 $count 条结果，你输入的是 Start=$Start End=$End"
    exit
}

# PowerShell 数组下标从 0 开始，所以要 -1
$selected = $scores[($Start - 1)..($End - 1)]

$avg = ($selected | Measure-Object -Average).Average

Write-Host "文件: $File"
Write-Host "总结果数: $count"
Write-Host "计算范围: 第 $Start 到第 $End 次"
Write-Host "样本数: $($selected.Count)"
Write-Host ("平均 weighted_total: {0:N2}" -f $avg)