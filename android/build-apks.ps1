param([switch]$InstallSdk)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$toolRoot = Join-Path $projectRoot '.cache\android-tools'
$toolPaths = Get-Content -LiteralPath (Join-Path $toolRoot 'paths.json') -Raw | ConvertFrom-Json
$env:JAVA_HOME = $toolPaths.java_home
$env:ANDROID_HOME = $toolPaths.sdk_root
$env:ANDROID_SDK_ROOT = $toolPaths.sdk_root
$env:GRADLE_USER_HOME = Join-Path $toolRoot 'gradle-cache'
$env:ANDROID_USER_HOME = Join-Path $toolRoot 'android-user'
$taskJavaHome = Join-Path $toolRoot 'java-user'
New-Item -ItemType Directory -Force -Path $env:ANDROID_USER_HOME, $taskJavaHome | Out-Null
$env:JAVA_TOOL_OPTIONS = "-Duser.home=$taskJavaHome"
$env:Path = (Join-Path $env:JAVA_HOME 'bin') + ';' + $env:Path
$sdkManager = Join-Path $env:ANDROID_HOME 'cmdline-tools\latest\bin\sdkmanager.bat'
if ($InstallSdk) {
    # Accept SDK terms for the requested build packages only.
    @(1..16 | ForEach-Object { 'y' }) | & $sdkManager --sdk_root=$env:ANDROID_HOME 'platforms;android-35' 'build-tools;35.0.0' 'platform-tools'
    if ($LASTEXITCODE -ne 0) { throw 'SDK setup failed.' }
}
$sdkProperty = $env:ANDROID_HOME.Replace('\','/')
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'local.properties'), "sdk.dir=$sdkProperty`n", [Text.UTF8Encoding]::new($false))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    & $python -m lab.prepare_android
    if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed.' }
} finally { Pop-Location }
Push-Location $PSScriptRoot
try {
    & $toolPaths.gradle --no-daemon --console=plain assembleRelease assembleReleaseAndroidTest
    if ($LASTEXITCODE -ne 0) { throw 'APK build failed.' }
} finally { Pop-Location }
$distribution = Join-Path $projectRoot 'artifacts\android'
New-Item -ItemType Directory -Force -Path $distribution | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'app\build\outputs\apk\release\app-release.apk') -Destination (Join-Path $distribution 'deploy-lab.apk')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'app\build\outputs\apk\androidTest\release\app-release-androidTest.apk') -Destination (Join-Path $distribution 'deploy-lab-test.apk')
Write-Output "APKs ready: $distribution"
