param(
    [string]$ProjectId = 'gggg-a7df3',
    [string]$Model = 'akita',
    [string]$AndroidApi = '34',
    [string]$ApkStoragePrefix = '',
    [ValidateSet(1,4)][int]$Threads = 1,
    [switch]$Submit
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$app = Join-Path $projectRoot 'artifacts\android\deploy-lab.apk'
$test = Join-Path $projectRoot 'artifacts\android\deploy-lab-test.apk'
if (-not (Test-Path -LiteralPath $app) -or -not (Test-Path -LiteralPath $test)) {
    throw 'Build both APKs first using android/build-apks.ps1.'
}
if ($ApkStoragePrefix) {
    if ($ApkStoragePrefix -notmatch '^gs://[^/\s]+/[^\s]+$') {
        throw 'ApkStoragePrefix must be a gs://bucket/folder URI without whitespace.'
    }
    $app = $ApkStoragePrefix.TrimEnd('/') + '/deploy-lab.apk'
    $test = $ApkStoragePrefix.TrimEnd('/') + '/deploy-lab-test.apk'
}
$gcloudCommand = Get-Command gcloud.cmd -ErrorAction Stop
$resultsDir = 'deploy-lab-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$arguments = @(
    'firebase', 'test', 'android', 'run', "--project=$ProjectId",
    '--type=instrumentation', "--app=$app", "--test=$test",
    "--device=model=$Model,version=$AndroidApi,locale=en,orientation=portrait",
    '--timeout=5m', "--environment-variables=threads=$Threads,warmup=30,runs=200",
    '--directories-to-pull=/sdcard/Android/data/com.vin.deploylab/files/benchmark',
    "--results-dir=$resultsDir", '--num-flaky-test-attempts=0',
    '--no-performance-metrics', '--no-record-video', '--no-use-orchestrator',
    '--no-auto-google-login',
    '--test-targets=class com.vin.deploylab.BenchmarkTest'
)
Write-Output "Project: $ProjectId; physical device requested: $Model; API: $AndroidApi; CPU threads: $Threads"
Write-Output "App: $app"
Write-Output "Test: $test"
Write-Output "Settings: FP32, batch 1, warm-up 30, 200 runs/runtime; results: $resultsDir"
if (-not $Submit) {
    Write-Output 'Preview only. No cloud test was submitted. Use -Submit after reviewing quota/cost.'
    exit 0
}
$deviceText = & $gcloudCommand.Source firebase test android models describe $Model "--project=$ProjectId" --format=json
if ($LASTEXITCODE -ne 0) { throw 'Cannot check the selected device.' }
$device = ($deviceText -join "`n") | ConvertFrom-Json
if ($device.form -ne 'PHYSICAL' -or $AndroidApi -notin $device.supportedVersionIds) {
    throw 'The selected model/API is not an available physical device configuration.'
}
& $gcloudCommand.Source @arguments
if ($LASTEXITCODE -ne 0) { throw "Test Lab returned exit code $LASTEXITCODE; inspect the test matrix and logs." }
