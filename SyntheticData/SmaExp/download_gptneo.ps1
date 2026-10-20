param(
    [string]$ModelRoot = 'D:\HOMEWORK\program\SyntheticData\model\gpt-neo-2.7B-8bit'
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
$base = 'https://hf-mirror.com/gustavecortal/gpt-neo-2.7B-8bit/resolve/main'
$tokenizerBase = 'https://hf-mirror.com/EleutherAI/gpt-neo-2.7B/resolve/main'

function Download-File([string]$Url, [string]$Path) {
    curl.exe -L --retry 5 --retry-delay 3 --continue-at - --output $Path $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
}

Download-File "$base/config.json" (Join-Path $ModelRoot 'config.json')
Download-File "$base/pytorch_model.bin" (Join-Path $ModelRoot 'pytorch_model.bin')
foreach ($name in @('merges.txt', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json')) {
    Download-File "$tokenizerBase/$name" (Join-Path $ModelRoot $name)
}
Write-Host "GPT-Neo 2.7B 8-bit files downloaded to $ModelRoot"
