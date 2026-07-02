param(
    [ValidateSet("audit", "telegram", "model")]
    [string]$Service,
    [switch]$Gpu,
    [string]$ComposeProjectName = "deepseek-fable",
    [string]$ApiKey = $env:DEEPSEEK_FABLE_API_KEY
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeArgs = @("--project-name", $ComposeProjectName)

if ($Gpu) {
    $composeArgs += @("-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")
}

$composeArgs += @("--profile", $Service)
$composeArgs += @("up", "-d", $Service)

$env:DEEPSEEK_FABLE_API_KEY = $ApiKey

docker compose @composeArgs
