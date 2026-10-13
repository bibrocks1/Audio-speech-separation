# Download + MD5-verify + extract LibriSpeech with built-in Windows tools only
# (curl.exe, Get-FileHash, tar.exe) - no Python needed. Resumable.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\download_librispeech.ps1 [-Root data\audio]
param([string]$Root = "data\audio", [string[]]$Subsets = @("dev-clean", "test-clean", "train-clean-100"))
$ErrorActionPreference = "Stop"
if (-not [System.IO.Path]::IsPathRooted($Root)) { $Root = Join-Path (Get-Location).Path $Root }
$md5 = @{ "dev-clean" = "42E2234BA48799C1F50F24A7926300A1"; "test-clean" = "32FA31D27D2E1CAD72775FEE3F4849A9";
          "train-clean-100" = "2A93770F6D5C6C964BC36631D331A522";
          "train-clean-360" = "C0E676E450A7FF2F54AEADE5171606FA" }
$mirrors = @("https://www.openslr.org/resources/12", "https://openslr.elda.org/resources/12", "https://openslr.trmal.net/resources/12")
$dl = Join-Path $Root "downloads"; New-Item -ItemType Directory -Force -Path $dl, (Join-Path $Root "LibriSpeech") | Out-Null
foreach ($s in $Subsets) {
    $marker = Join-Path $Root "LibriSpeech\.$s.done"
    if (Test-Path $marker) { Write-Output "[skip] $s already extracted"; continue }
    $free = [math]::Round((Get-PSDrive ($Root.Substring(0,1))).Free / 1GB, 1)
    Write-Output "[$s] free disk: $free GB"
    $arc = Join-Path $dl "$s.tar.gz"
    $ok = (Test-Path $arc) -and ((Get-FileHash -Algorithm MD5 $arc).Hash -eq $md5[$s])
    foreach ($m in $mirrors) {
        if ($ok) { break }
        Write-Output "[$s] downloading $m/$s.tar.gz  $(Get-Date -Format T)"
        & curl.exe -L -sS -C - --retry 10 --retry-delay 15 -o $arc "$m/$s.tar.gz"
        if ($LASTEXITCODE -ne 0) { Write-Output "[$s] curl exit $LASTEXITCODE from $m"; continue }
        $h = (Get-FileHash -Algorithm MD5 $arc).Hash
        $ok = $h -eq $md5[$s]
        Write-Output "[$s] md5 $h $(if ($ok) {'OK'} else {'MISMATCH, expected ' + $md5[$s]})"
        if (-not $ok) { Move-Item -Force $arc "$arc.bad" }
    }
    if (-not $ok) { Write-Output "[$s] FAILED"; exit 1 }
    Write-Output "[$s] extracting  $(Get-Date -Format T)"
    & tar.exe -xzf $arc -C $Root
    if ($LASTEXITCODE -ne 0) { Write-Output "[$s] tar exit $LASTEXITCODE"; exit 1 }
    Set-Content -Path $marker -Value "ok"
    $n = (Get-ChildItem -Recurse -Filter *.flac (Join-Path $Root "LibriSpeech\$s")).Count
    Write-Output "[$s] done: $n flac files  $(Get-Date -Format T)"
}
