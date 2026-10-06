[CmdletBinding()]
param(
    [ValidateRange(1, 720)]
    [int]$MaxMinutes = 240
)

$ErrorActionPreference = 'Stop'

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class DefectAugPowerState
{
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@

$esContinuous = [Convert]::ToUInt32('80000000', 16)
$esSystemRequired = [uint32]0x00000001
$deadline = [DateTimeOffset]::Now.AddMinutes($MaxMinutes)

try {
    $result = [DefectAugPowerState]::SetThreadExecutionState($esContinuous -bor $esSystemRequired)
    if ($result -eq 0) {
        throw 'SetThreadExecutionState failed.'
    }
    Write-Output "Temporary system-sleep prevention active until $($deadline.ToString('o')) or process exit. Display sleep is unchanged."
    while ([DateTimeOffset]::Now -lt $deadline) {
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][DefectAugPowerState]::SetThreadExecutionState($esContinuous)
    Write-Output 'Temporary system-sleep prevention released.'
}
