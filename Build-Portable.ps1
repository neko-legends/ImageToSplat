$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

python -m PyInstaller --noconfirm --clean .\NekoSplatForge.spec

$distPortableRoot = Join-Path $projectRoot "dist\NekoSplatForge"
$portableRoot = Join-Path $projectRoot "release\portable"
$portablePackageRoot = Join-Path $portableRoot "NekoSplatForge"

if (-not (Test-Path -LiteralPath $distPortableRoot)) {
  throw "PyInstaller did not create the expected portable folder: $distPortableRoot"
}

$resolvedPortableRoot = [System.IO.Path]::GetFullPath($portableRoot)
$resolvedPortablePackageRoot = [System.IO.Path]::GetFullPath($portablePackageRoot)
if (-not $resolvedPortablePackageRoot.StartsWith($resolvedPortableRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to remove portable package outside release\portable: $portablePackageRoot"
}

New-Item -ItemType Directory -Force -Path $portableRoot | Out-Null
if (Test-Path -LiteralPath $portablePackageRoot) {
  Remove-Item -LiteralPath $portablePackageRoot -Recurse -Force
}
Copy-Item -LiteralPath $distPortableRoot -Destination $portableRoot -Recurse -Force

$ckptRoot = Join-Path $portablePackageRoot "ckpts"
New-Item -ItemType Directory -Force -Path (Join-Path $ckptRoot "diffusion_models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ckptRoot "vae") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ckptRoot "clip_vision") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ckptRoot "background_removal") | Out-Null

@"
Neko Legends Splat Forge Portable

Run:
  NekoSplatForge.exe

Model weights:
  Launch the app and use the Setup popup to download the TripoSplat files.
  Files are installed into this ckpts folder.

Expected files:
  ckpts\diffusion_models\triposplat_fp16.safetensors
  ckpts\vae\triposplat_vae_decoder_fp16.safetensors
  ckpts\clip_vision\dino_v3_vit_h.safetensors
  ckpts\vae\flux2-vae.safetensors
  ckpts\background_removal\birefnet.safetensors

Optional settings:
  TRIPOSPLAT_PORT=7861
  TRIPOSPLAT_HOST=127.0.0.1
  TRIPOSPLAT_DEVICE=cuda
  TRIPOSPLAT_OPEN_BROWSER=0
  TRIPOSPLAT_AGENT_API=1
  TRIPOSPLAT_AGENT_API_PORT=17340
  TRIPOSPLAT_HEADLESS=1

Agent API:
  NekoSplatForge.exe --agent-api --no-browser
  NekoSplatForge.exe --serve-agent-api
  GET http://127.0.0.1:17340/openapi.json

Headless generation:
  NekoSplatForge.exe --headless input.png --output-dir output --output-name splat
"@ | Set-Content -Path (Join-Path $portablePackageRoot "README_PORTABLE.txt") -Encoding UTF8

Write-Host "Portable build created:"
Write-Host "  $portablePackageRoot"
Write-Host "Executable:"
Write-Host "  $(Join-Path $portablePackageRoot 'NekoSplatForge.exe')"
