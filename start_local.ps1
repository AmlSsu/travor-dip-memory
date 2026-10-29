param(
    [int]$Port = 18098,
    [string[]]$EnvFile,
    [string]$PythonPath,
    [ValidateSet('dashscope','bge')][string]$EmbeddingBackend = 'dashscope',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [ValidateSet('hybrid_window','hybrid','adaptive','dense','evidence_window','stem_window','focus_stem_rules','query_document_evidence')][string]$Mode = 'query_document_evidence',
    [int]$ContextChars = 24000
)
$ErrorActionPreference = 'Stop'
foreach ($configPath in $EnvFile) {
    foreach ($line in Get-Content -LiteralPath $configPath -Encoding UTF8) {
        if ($line -match '^\s*(#|$)') { continue }
        if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { throw 'Invalid environment file line.' }
        $settingName = $Matches[1]
        $settingValue = $Matches[2].Trim()
        if (($settingValue.StartsWith('"') -and $settingValue.EndsWith('"')) -or ($settingValue.StartsWith("'") -and $settingValue.EndsWith("'"))) {
            $settingValue = $settingValue.Substring(1, $settingValue.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($settingName, $settingValue, 'Process')
    }
}
$projectPath = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $PSScriptRoot '.venv-cycle2\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = 'python' }
}
$env:AE_EMBEDDING_BACKEND = $EmbeddingBackend
if ($EmbeddingBackend -eq 'bge') {
    $env:AE_EMBEDDING_PATH = Join-Path $projectPath 'MemoryBench-Shared\models\embeddings\bge-small-en-v1.5'
} elseif (-not $env:DASHSCOPE_API_KEY -or -not $env:AE_EMBEDDING_URL) {
    throw 'Set DASHSCOPE_API_KEY and AE_EMBEDDING_URL before starting the Cycle 2 service.'
}
if (-not $env:AE_DATABASE) { throw 'Set an explicit, separate AE_DATABASE for this model profile.' }
if (-not [IO.Path]::IsPathRooted($env:AE_DATABASE)) { $env:AE_DATABASE = Join-Path $PSScriptRoot $env:AE_DATABASE }
$env:AE_DATABASE = [IO.Path]::GetFullPath($env:AE_DATABASE)
# This release has no generative Add/Search implementation, regardless of inherited flags.
$env:AE_ADD_INDEXER = '0'
$env:AE_PLANNER = '0'
$env:AE_LOCAL_EVIDENCE_SEARCH = '0'
$env:AE_LOCAL_EVIDENCE_EXPAND = '0'
$env:AE_DEVICE = $Device
if ($PSBoundParameters.ContainsKey('Mode') -or -not $env:AE_MODE) { $env:AE_MODE = $Mode }
if ($PSBoundParameters.ContainsKey('ContextChars') -or -not $env:AE_CONTEXT_CHARS) { $env:AE_CONTEXT_CHARS = [string]$ContextChars }
$env:PYTHONPATH = $PSScriptRoot
if (-not $env:MEMORY_API_KEY) { throw 'Set MEMORY_API_KEY before starting the service.' }
# Windows PowerShell treats uvicorn's normal stderr startup log as an error
# record. Keep configuration validation strict, but let the native exit code
# determine server failure.
$ErrorActionPreference = 'Continue'
& $pythonPath -m uvicorn adaptive_evidence.api:app --host 127.0.0.1 --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
