<#
.SYNOPSIS
Safely plan, download, or verify the four Qwen-Image inpainting model files used by
the ComfyUI Image Inpainting (Qwen-image) blueprint.

.DESCRIPTION
The default mode is Plan and performs no writes. Download mode requires the
explicit -ConfirmLargeDownload switch. Every source URL is pinned to a Hugging
Face repository revision and every file is checked against a fixed byte length
and SHA-256 digest.

.PARAMETER Mode
Plan (default), Download, or Verify.

.PARAMETER ConfirmLargeDownload
Required with -Mode Download. The four files total about 31.95 GiB.

.PARAMETER ComfyRoot
Absolute path to the active ComfyUI root. The directory must contain main.py and
models.

.PARAMETER Aria2Path
Absolute path to aria2c.exe.

.PARAMETER ProxyUrl
HTTP(S) proxy used by aria2. Download mode checks that its TCP endpoint is open.

.PARAMETER ReserveFreeGiB
Minimum free space that must remain on the destination drive after the expected
remaining bytes are downloaded.

.EXAMPLE
pwsh -NoProfile -File .\scripts\download_qwen_inpaint_models.ps1 -Mode Plan

.EXAMPLE
pwsh -NoProfile -File .\scripts\download_qwen_inpaint_models.ps1 -Mode Download -ConfirmLargeDownload

.EXAMPLE
pwsh -NoProfile -File .\scripts\download_qwen_inpaint_models.ps1 -Mode Verify
#>

[CmdletBinding()]
param(
    [ValidateSet("Plan", "Download", "Verify")]
    [string]$Mode = "Plan",

    [switch]$ConfirmLargeDownload,

    [ValidateSet("Curl", "Aria2")]
    [string]$Downloader = "Curl",

    [ValidateSet("HuggingFace", "ModelScope")]
    [string]$DownloadSource = "HuggingFace",

    [string]$ComfyRoot = "D:\ComfyUI-aki\ComfyUI-aki-v1.6\ComfyUI",

    [string]$Aria2Path = "D:\ComfyUI-aki\ComfyUI-aki-v1.6\.launcher\aria2c.exe",

    [string]$CurlPath = "C:\Windows\System32\curl.exe",

    [string]$ProxyUrl = "http://127.0.0.1:7890",

    [ValidateRange(0, 1024)]
    [int]$ReserveFreeGiB = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$reportsRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "reports"))
$manifestJsonPath = Join-Path $reportsRoot "qwen_inpaint_models_manifest.json"
$manifestShaPath = Join-Path $reportsRoot "qwen_inpaint_models.sha256"

$qwenRevision = "7beb7b647f04469fbe64ba8adc2bb0d7e5e9f73f"
$controlNetRevision = "14281e338504b697e300467af940e4fbbeea4cdf"
$modelScopeQwenRevision = "ce621e1585ef9720cfcf641c741f350394aad681"
$modelScopeVaeRevision = "350816cfd6687d47c7a39eff16c1f339ae531687"
$modelScopeControlNetRevision = "a18ddf38dce2af72f0306db92baeaab4b84c3cf7"
$useModelScope = $DownloadSource -eq "ModelScope"

$models = @(
    [pscustomobject]@{
        Name = "qwen_image_fp8_e4m3fn.safetensors"
        RelativeDirectory = "diffusion_models"
        ExpectedBytes = [long]20430635136
        ExpectedSha256 = "98763a127701eb6fb59096f7742cb3aa7d64ed510b9f4e882d8351f8176e3ce3"
        SourceRepo = "Comfy-Org/Qwen-Image_ComfyUI"
        UpstreamRevision = $qwenRevision
        SourceRevision = $(if ($useModelScope) { $modelScopeQwenRevision } else { $qwenRevision })
        SourceUrl = $(if ($useModelScope) { "https://modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/$modelScopeQwenRevision/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" } else { "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/$qwenRevision/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" })
    },
    [pscustomobject]@{
        Name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        RelativeDirectory = "text_encoders"
        ExpectedBytes = [long]9384670680
        ExpectedSha256 = "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"
        SourceRepo = "Comfy-Org/Qwen-Image_ComfyUI"
        UpstreamRevision = $qwenRevision
        SourceRevision = $(if ($useModelScope) { $modelScopeQwenRevision } else { $qwenRevision })
        SourceUrl = $(if ($useModelScope) { "https://modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/$modelScopeQwenRevision/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" } else { "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/$qwenRevision/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" })
    },
    [pscustomobject]@{
        Name = "qwen_image_vae.safetensors"
        RelativeDirectory = "vae"
        ExpectedBytes = [long]253806246
        ExpectedSha256 = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
        SourceRepo = "Comfy-Org/Qwen-Image_ComfyUI"
        UpstreamRevision = $qwenRevision
        SourceRevision = $(if ($useModelScope) { $modelScopeVaeRevision } else { $qwenRevision })
        SourceUrl = $(if ($useModelScope) { "https://modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/$modelScopeVaeRevision/split_files/vae/qwen_image_vae.safetensors" } else { "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/$qwenRevision/split_files/vae/qwen_image_vae.safetensors" })
    },
    [pscustomobject]@{
        Name = "Qwen-Image-InstantX-ControlNet-Inpainting.safetensors"
        RelativeDirectory = "controlnet"
        ExpectedBytes = [long]4234599432
        ExpectedSha256 = "49c01aafe6545c1f6e7627a724c8fb13357fe74efee235622158f0a8f30e5458"
        SourceRepo = "Comfy-Org/Qwen-Image-InstantX-ControlNets"
        UpstreamRevision = $controlNetRevision
        SourceRevision = $(if ($useModelScope) { $modelScopeControlNetRevision } else { $controlNetRevision })
        SourceUrl = $(if ($useModelScope) { "https://modelscope.cn/models/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/$modelScopeControlNetRevision/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors" } else { "https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/$controlNetRevision/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors" })
    }
)

function Get-DriveQualifiedFullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not [IO.Path]::IsPathRooted($Path)) {
        throw "$Label must be an absolute path: $Path"
    }

    $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath -notmatch "^[A-Za-z]:\\") {
        throw "$Label must be drive-qualified on this Windows host: $fullPath"
    }
    return $fullPath
}

function Assert-PathInsideRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Candidate,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $separator = [IO.Path]::DirectorySeparatorChar
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd("\", "/") + $separator
    $candidateFull = [IO.Path]::GetFullPath($Candidate)
    if (-not $candidateFull.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes the approved root. Root=$rootFull Candidate=$candidateFull"
    }
}

function Assert-NotReparsePoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label must not be a symlink or reparse point: $Path"
        }
    }
}

function Get-ModelState {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$Model,

        [Parameter(Mandatory = $true)]
        [string]$ModelsRoot
    )

    $targetDirectory = [IO.Path]::GetFullPath((Join-Path $ModelsRoot $Model.RelativeDirectory))
    $targetPath = [IO.Path]::GetFullPath((Join-Path $targetDirectory $Model.Name))
    Assert-PathInsideRoot -Root $ModelsRoot -Candidate $targetDirectory -Label "Model directory"
    Assert-PathInsideRoot -Root $ModelsRoot -Candidate $targetPath -Label "Model file"
    Assert-NotReparsePoint -Path $targetDirectory -Label "Model directory"
    Assert-NotReparsePoint -Path $targetPath -Label "Model file"

    $status = "Missing"
    $actualBytes = [long]0
    $actualSha256 = $null

    if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
        $file = Get-Item -LiteralPath $targetPath -Force
        $actualBytes = [long]$file.Length
        if ($actualBytes -lt $Model.ExpectedBytes) {
            $status = "Partial"
        }
        elseif ($actualBytes -gt $Model.ExpectedBytes) {
            $status = "UnexpectedSize"
        }
        else {
            Write-Host "Hashing existing file: $targetPath"
            $actualSha256 = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualSha256 -eq $Model.ExpectedSha256) {
                $status = "Complete"
            }
            else {
                $status = "HashMismatch"
            }
        }
    }

    return [pscustomobject]@{
        Model = $Model
        Name = $Model.Name
        RelativePath = (Join-Path $Model.RelativeDirectory $Model.Name)
        TargetDirectory = $targetDirectory
        TargetPath = $targetPath
        Status = $status
        ExpectedBytes = [long]$Model.ExpectedBytes
        ActualBytes = $actualBytes
        ExpectedSha256 = $Model.ExpectedSha256
        ActualSha256 = $actualSha256
        SourceRepo = $Model.SourceRepo
        SourceRevision = $Model.SourceRevision
        UpstreamRevision = $Model.UpstreamRevision
        SourceUrl = $Model.SourceUrl
    }
}

function Test-ProxyEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [Uri]$ProxyUri,

        [int]$TimeoutMilliseconds = 3000
    )

    $client = New-Object Net.Sockets.TcpClient
    $asyncResult = $null
    try {
        $asyncResult = $client.BeginConnect($ProxyUri.Host, $ProxyUri.Port, $null, $null)
        if (-not $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false)) {
            throw "Proxy TCP check timed out: $($ProxyUri.Host):$($ProxyUri.Port)"
        }
        $client.EndConnect($asyncResult)
    }
    finally {
        if ($null -ne $asyncResult) {
            $asyncResult.AsyncWaitHandle.Close()
        }
        $client.Close()
    }
}

function Invoke-ModelDownload {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$State,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedAria2Path,

        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [string]$ResolvedProxyUrl
    )

    if ($State.Status -eq "Complete") {
        Write-Host "SKIP complete: $($State.TargetPath)"
        return
    }
    if ($State.Status -in @("UnexpectedSize", "HashMismatch")) {
        throw "Refusing to overwrite suspicious file ($($State.Status)): $($State.TargetPath). Move it aside manually, then rerun."
    }

    if (-not (Test-Path -LiteralPath $State.TargetDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $State.TargetDirectory | Out-Null
    }
    Assert-NotReparsePoint -Path $State.TargetDirectory -Label "Model directory"

    if ($State.Status -eq "Partial") {
        Write-Host "RESUME partial: $($State.TargetPath) ($($State.ActualBytes)/$($State.ExpectedBytes) bytes)"
    }
    else {
        Write-Host "DOWNLOAD: $($State.TargetPath)"
    }

    $aria2Arguments = @(
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--file-allocation=none",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=16M",
        "--connect-timeout=30",
        "--timeout=60",
        "--max-tries=10",
        "--retry-wait=5",
        "--check-certificate=true",
        "--check-integrity=true",
        "--checksum=sha-256=$($State.ExpectedSha256)",
        "--summary-interval=10",
        "--console-log-level=notice",
        "--dir=$($State.TargetDirectory)",
        "--out=$($State.Name)",
        $State.SourceUrl
    )
    if ($ResolvedProxyUrl) {
        $aria2Arguments = @("--all-proxy=$ResolvedProxyUrl") + $aria2Arguments
    }

    & $ResolvedAria2Path @aria2Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "aria2 exited with code $LASTEXITCODE while downloading $($State.Name)"
    }
}

function Invoke-CurlModelDownload {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$State,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedCurlPath,

        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [string]$ResolvedProxyUrl
    )

    if ($State.Status -eq "Complete") {
        Write-Host "SKIP complete: $($State.TargetPath)"
        return
    }
    if ($State.Status -in @("UnexpectedSize", "HashMismatch")) {
        throw "Refusing to overwrite suspicious file ($($State.Status)): $($State.TargetPath). Move it aside manually, then rerun."
    }

    if (-not (Test-Path -LiteralPath $State.TargetDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $State.TargetDirectory | Out-Null
    }
    Assert-NotReparsePoint -Path $State.TargetDirectory -Label "Model directory"

    if ($State.Status -eq "Partial") {
        Write-Host "RESUME partial with curl: $($State.TargetPath) ($($State.ActualBytes)/$($State.ExpectedBytes) bytes)"
    }
    else {
        Write-Host "DOWNLOAD with curl: $($State.TargetPath)"
    }

    $curlArguments = @(
        "--location",
        "--fail",
        "--continue-at", "-",
        "--connect-timeout", "30",
        "--max-time", "0",
        "--output", $State.TargetPath,
        "--show-error",
        "--progress-bar",
        $State.SourceUrl
    )
    if ($ResolvedProxyUrl) {
        $curlArguments = @("--proxy", $ResolvedProxyUrl) + $curlArguments
    }

    # Do not use curl's built-in --retry with --continue-at and the same output
    # file.  On this Windows/proxy combination an in-process retry can reopen at
    # the offset that existed when curl first started, discarding bytes obtained
    # by the failed attempt.  Restarting curl here makes every attempt discover
    # the latest on-disk length.  The monotonic-size check fails closed if a
    # transport ever truncates the partial file.
    $maxAttempts = 20
    $completed = $false
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $beforeBytes = if (Test-Path -LiteralPath $State.TargetPath -PathType Leaf) {
            [long](Get-Item -LiteralPath $State.TargetPath -Force).Length
        }
        else {
            [long]0
        }
        Write-Host "curl attempt $attempt/$maxAttempts starting at $beforeBytes bytes"
        & $ResolvedCurlPath @curlArguments
        $curlExitCode = $LASTEXITCODE
        $afterBytes = if (Test-Path -LiteralPath $State.TargetPath -PathType Leaf) {
            [long](Get-Item -LiteralPath $State.TargetPath -Force).Length
        }
        else {
            [long]0
        }
        if ($afterBytes -lt $beforeBytes) {
            throw "curl truncated the partial file: $beforeBytes -> $afterBytes bytes for $($State.Name)"
        }
        if ($curlExitCode -eq 0) {
            $completed = $true
            break
        }
        if ($attempt -lt $maxAttempts) {
            Write-Warning "curl exited with code $curlExitCode after reaching $afterBytes bytes; retrying from the latest length in 5 seconds."
            Start-Sleep -Seconds 5
        }
    }
    if (-not $completed) {
        throw "curl did not complete $($State.Name) after $maxAttempts attempts"
    }
}

function Write-VerificationManifests {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$States,

        [Parameter(Mandatory = $true)]
        [object[]]$InitialStates,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedComfyRoot
    )

    Assert-PathInsideRoot -Root $projectRoot -Candidate $reportsRoot -Label "Reports directory"
    Assert-PathInsideRoot -Root $projectRoot -Candidate $manifestJsonPath -Label "JSON manifest"
    Assert-PathInsideRoot -Root $projectRoot -Candidate $manifestShaPath -Label "SHA-256 manifest"
    Assert-NotReparsePoint -Path $reportsRoot -Label "Reports directory"

    if (-not (Test-Path -LiteralPath $reportsRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $reportsRoot | Out-Null
    }

    $initialByName = @{}
    foreach ($initialState in $InitialStates) {
        $initialByName[$initialState.Name] = $initialState
    }
    $priorManifest = $null
    $priorByRelativePath = @{}
    if (Test-Path -LiteralPath $manifestJsonPath -PathType Leaf) {
        try {
            $priorManifest = Get-Content -LiteralPath $manifestJsonPath -Raw | ConvertFrom-Json
            foreach ($priorFile in @($priorManifest.files)) {
                if ($priorFile.relative_path) {
                    $priorByRelativePath[[string]$priorFile.relative_path] = $priorFile
                }
            }
        }
        catch {
            throw "Existing Qwen model manifest is unreadable; refusing to overwrite provenance: $manifestJsonPath"
        }
    }

    $manifest = [ordered]@{
        schema_version = 1
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        comfy_root = $ResolvedComfyRoot
        selected_download_source = $(if ($priorManifest -and $priorManifest.selected_download_source) { [string]$priorManifest.selected_download_source } else { $DownloadSource })
        provenance_note = "initial_status and initial_bytes distinguish pre-existing, resumed, and newly downloaded content; final SHA-256 is authoritative for byte identity"
        source_revisions = [ordered]@{
            "Comfy-Org/Qwen-Image_ComfyUI" = $qwenRevision
            "Comfy-Org/Qwen-Image-InstantX-ControlNets" = $controlNetRevision
        }
        files = @(
            $States | ForEach-Object {
                $initial = $initialByName[$_.Name]
                $relativePath = $_.RelativePath.Replace("\", "/")
                $prior = $priorByRelativePath[$relativePath]
                $preservePrior = $prior -and ([string]$prior.sha256).ToLowerInvariant() -eq $_.ActualSha256
                $disposition = if ($preservePrior -and $prior.transfer_disposition) {
                    [string]$prior.transfer_disposition
                }
                elseif ($initial.Status -eq "Complete") {
                    "preexisting_verified_by_hash"
                }
                elseif ($initial.Status -eq "Partial") {
                    "resumed_via_$DownloadSource"
                }
                else {
                    "downloaded_via_$DownloadSource"
                }
                [ordered]@{
                    relative_path = $relativePath
                    bytes = [long]$_.ActualBytes
                    sha256 = $_.ActualSha256
                    expected_sha256 = $_.ExpectedSha256
                    initial_status = $(if ($preservePrior -and $prior.initial_status) { [string]$prior.initial_status } else { $initial.Status })
                    initial_bytes = $(if ($preservePrior -and $null -ne $prior.initial_bytes) { [long]$prior.initial_bytes } else { [long]$initial.ActualBytes })
                    transfer_disposition = $disposition
                    source_repo = $(if ($preservePrior -and $prior.source_repo) { [string]$prior.source_repo } else { $_.SourceRepo })
                    source_revision = $(if ($preservePrior -and $prior.source_revision) { [string]$prior.source_revision } else { $_.SourceRevision })
                    upstream_revision = $(if ($preservePrior -and $prior.upstream_revision) { [string]$prior.upstream_revision } else { $_.UpstreamRevision })
                    source_url = $(if ($preservePrior -and $prior.source_url) { [string]$prior.source_url } else { $_.SourceUrl })
                }
            }
        )
    }

    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestJsonPath -Encoding UTF8
    $shaLines = $States | ForEach-Object {
        "{0} *{1}" -f $_.ActualSha256, $_.RelativePath.Replace("\", "/")
    }
    $shaLines | Set-Content -LiteralPath $manifestShaPath -Encoding ASCII

    Write-Host "Wrote manifest: $manifestJsonPath"
    Write-Host "Wrote SHA-256 list: $manifestShaPath"
}

$resolvedComfyRoot = Get-DriveQualifiedFullPath -Path $ComfyRoot -Label "ComfyRoot"
$resolvedAria2Path = $null
$resolvedCurlPath = $null
if ($Downloader -eq "Aria2") {
    $resolvedAria2Path = Get-DriveQualifiedFullPath -Path $Aria2Path -Label "Aria2Path"
}
else {
    $resolvedCurlPath = Get-DriveQualifiedFullPath -Path $CurlPath -Label "CurlPath"
}

if (-not (Test-Path -LiteralPath $resolvedComfyRoot -PathType Container)) {
    throw "ComfyUI root does not exist: $resolvedComfyRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedComfyRoot "main.py") -PathType Leaf)) {
    throw "ComfyUI main.py was not found under: $resolvedComfyRoot"
}

$modelsRoot = [IO.Path]::GetFullPath((Join-Path $resolvedComfyRoot "models"))
if (-not (Test-Path -LiteralPath $modelsRoot -PathType Container)) {
    throw "ComfyUI models directory does not exist: $modelsRoot"
}
Assert-PathInsideRoot -Root $resolvedComfyRoot -Candidate $modelsRoot -Label "Models root"
Assert-NotReparsePoint -Path $modelsRoot -Label "Models root"

if ($Downloader -eq "Aria2" -and -not (Test-Path -LiteralPath $resolvedAria2Path -PathType Leaf)) {
    throw "aria2c.exe does not exist: $resolvedAria2Path"
}
if ($Downloader -eq "Curl" -and -not (Test-Path -LiteralPath $resolvedCurlPath -PathType Leaf)) {
    throw "curl.exe does not exist: $resolvedCurlPath"
}
if ($Downloader -eq "Aria2") {
    Assert-NotReparsePoint -Path $resolvedAria2Path -Label "aria2c.exe"
}
else {
    Assert-NotReparsePoint -Path $resolvedCurlPath -Label "curl.exe"
}

$proxyUri = $null
$effectiveProxyUrl = $null
if (-not $useModelScope) {
    try {
        $proxyUri = [Uri]$ProxyUrl
    }
    catch {
        throw "ProxyUrl is not a valid URI: $ProxyUrl"
    }
    if (-not $proxyUri.IsAbsoluteUri -or $proxyUri.Scheme -notin @("http", "https")) {
        throw "ProxyUrl must be an absolute HTTP(S) URI: $ProxyUrl"
    }
    $effectiveProxyUrl = $ProxyUrl
}

$states = @($models | ForEach-Object { Get-ModelState -Model $_ -ModelsRoot $modelsRoot })

Write-Host "Mode: $Mode"
Write-Host "ComfyUI root: $resolvedComfyRoot"
Write-Host "Downloader: $Downloader"
Write-Host "Download source: $DownloadSource"
if ($Downloader -eq "Aria2") {
    Write-Host "aria2: $resolvedAria2Path"
}
else {
    Write-Host "curl: $resolvedCurlPath"
}
Write-Host "Proxy: $(if ($effectiveProxyUrl) { $effectiveProxyUrl } else { 'direct' })"
$states | Select-Object Status, Name, ActualBytes, ExpectedBytes, TargetPath | Format-Table -AutoSize

$remainingBytes = [long]0
foreach ($state in $states) {
    if ($state.Status -eq "Partial") {
        $remainingBytes += ($state.ExpectedBytes - $state.ActualBytes)
    }
    elseif ($state.Status -ne "Complete") {
        $remainingBytes += $state.ExpectedBytes
    }
}

$driveRoot = [IO.Path]::GetPathRoot($modelsRoot)
$driveInfo = New-Object IO.DriveInfo($driveRoot)
$reserveBytes = [long]$ReserveFreeGiB * 1GB
$projectedFreeBytes = [long]$driveInfo.AvailableFreeSpace - $remainingBytes
Write-Host ("Remaining download: {0:N3} GiB" -f ($remainingBytes / 1GB))
Write-Host ("Drive free now: {0:N3} GiB; projected free: {1:N3} GiB; required reserve: {2} GiB" -f ($driveInfo.AvailableFreeSpace / 1GB), ($projectedFreeBytes / 1GB), $ReserveFreeGiB)

if ($Mode -eq "Plan") {
    Write-Host "PLAN ONLY: no directories, model files, or manifests were changed."
    return
}

if ($Mode -eq "Download") {
    if (-not $ConfirmLargeDownload) {
        throw "Download mode requires -ConfirmLargeDownload. Run Plan first."
    }
    if ($states.Status -contains "UnexpectedSize" -or $states.Status -contains "HashMismatch") {
        throw "Suspicious existing files were found. No download started. Move those files aside manually and rerun Plan."
    }
    if ($projectedFreeBytes -lt $reserveBytes) {
        throw "Insufficient free space. The operation would leave less than $ReserveFreeGiB GiB free."
    }

    if ($effectiveProxyUrl) {
        Write-Host "Checking proxy TCP endpoint..."
        Test-ProxyEndpoint -ProxyUri $proxyUri
        Write-Host "Proxy endpoint is reachable."
    }

    foreach ($state in $states) {
        if ($Downloader -eq "Aria2") {
            Invoke-ModelDownload -State $state -ResolvedAria2Path $resolvedAria2Path -ResolvedProxyUrl $effectiveProxyUrl
        }
        else {
            Invoke-CurlModelDownload -State $state -ResolvedCurlPath $resolvedCurlPath -ResolvedProxyUrl $effectiveProxyUrl
        }
    }
}

$finalStates = @($models | ForEach-Object { Get-ModelState -Model $_ -ModelsRoot $modelsRoot })
$notComplete = @($finalStates | Where-Object { $_.Status -ne "Complete" })
if ($notComplete.Count -gt 0) {
    $summary = ($notComplete | ForEach-Object { "$($_.Name)=$($_.Status)" }) -join ", "
    throw "Verification failed; manifests were not written: $summary"
}

Write-VerificationManifests -States $finalStates -InitialStates $states -ResolvedComfyRoot $resolvedComfyRoot
Write-Host "All four model files match their pinned byte lengths and SHA-256 digests."
