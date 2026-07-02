param(
    [ValidateSet("serve", "audit", "telegram")]
    [string]$Mode = "serve",
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000,
    [string]$ModelPath = "",
    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Device = "auto",
    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$DType = "auto",
    [string]$ApiKey = $env:DEEPSEEK_FABLE_API_KEY
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "未找到虚拟环境，请先重新创建 .venv"
}

$env:DEEPSEEK_FABLE_API_KEY = $ApiKey

switch ($Mode) {
    "serve" {
        if (-not $ModelPath) {
            throw "Mode=serve 时必须提供 -ModelPath"
        }
        & $venvPython -m deepseek_fable_lab.cli serve --model-path $ModelPath --host $Host --port $Port --device $Device --dtype $DType
    }
    "audit" {
        & $venvPython -m deepseek_fable_lab.cli audit --host $Host --port $Port --api-key-env DEEPSEEK_FABLE_API_KEY
    }
    "telegram" {
        & $venvPython -m deepseek_fable_lab.cli telegram --host $Host --port $Port --api-key-env DEEPSEEK_FABLE_API_KEY
    }
}
