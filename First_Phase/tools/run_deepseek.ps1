param(
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$OutputFile
)

$ErrorActionPreference = 'Stop'
$environmentNames = @('ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN')
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$deepseekExitCode = 1
try {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $taskPrompt = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $PromptFile).Path, $utf8)
    $credentialText = [System.IO.File]::ReadAllText('D:\FYP\deepseek_api_key.txt', $utf8)
    $credentialMatches = [regex]::Matches($credentialText, 'sk-[A-Za-z0-9_-]+')
    if ($credentialMatches.Count -ne 1) { throw ('DeepSeek credential matches: ' + $credentialMatches.Count) }
    $credential = $credentialMatches[0].Value
    [Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', $credential, 'Process')
    [Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', 'https://api.deepseek.com/anthropic', 'Process')
    [Environment]::SetEnvironmentVariable('ANTHROPIC_AUTH_TOKEN', $null, 'Process')
    $credential = $null
    $credentialText = $null
    $credentialMatches = $null

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = (Get-Command claude.exe -ErrorAction Stop).Source
    $startInfo.WorkingDirectory = 'D:\FYP'
    $startInfo.Arguments = '--bare -p --model deepseek-flash --setting-sources "" --no-session-persistence --output-format json --allowedTools Read Write Edit Bash --tools Read,Write,Edit,Bash'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = $utf8
    $startInfo.StandardErrorEncoding = $utf8
    if ($startInfo.PSObject.Properties.Name -contains 'StandardInputEncoding') {
        $startInfo.StandardInputEncoding = $utf8
    }
    $deepseekProcess = New-Object System.Diagnostics.Process
    $deepseekProcess.StartInfo = $startInfo
    [void]$deepseekProcess.Start()
    $stdoutTask = $deepseekProcess.StandardOutput.ReadToEndAsync()
    $stderrTask = $deepseekProcess.StandardError.ReadToEndAsync()
    $deepseekProcess.StandardInput.WriteLine($taskPrompt)
    $deepseekProcess.StandardInput.Close()
    $deepseekProcess.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $deepseekExitCode = $deepseekProcess.ExitCode
    [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputFile), $stdout, $utf8)
    if (-not [string]::IsNullOrEmpty($stderr)) {
        [System.IO.File]::AppendAllText([System.IO.Path]::GetFullPath($OutputFile), "`nSTDERR`n" + $stderr, $utf8)
        [Console]::Error.Write($stderr)
    }
    [Console]::Write($stdout)
    $deepseekProcess.Dispose()
}
catch {
    [Console]::Error.WriteLine($_.Exception.ToString())
    $deepseekExitCode = 1
}
finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
}
exit $deepseekExitCode
