$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$buildScript = Join-Path $projectRoot "scripts\build_pympress_notes_pdf.ps1"
$notesPdf = Join-Path $projectRoot "gnn_bandgap_research_final_slides_pympress.pdf"
$plainPdf = Join-Path $projectRoot "gnn_bandgap_research_final_slides.pdf"
$pympress = Join-Path $env:LOCALAPPDATA "Programs\pympress-1.8.6-x86_64.msi\pympress.exe"

if (-not (Test-Path $pympress)) {
    throw "Pympress executable not found: $pympress"
}

if (-not (Test-Path $buildScript)) {
    throw "Build script not found: $buildScript"
}

& $buildScript

if (Test-Path $notesPdf) {
    Start-Process -FilePath $pympress -ArgumentList @("-q") -Wait
    Start-Sleep -Milliseconds 500
    Start-Process -FilePath $pympress -ArgumentList @("-N", "right", "-t", "8", $notesPdf)
    return
}

if (-not (Test-Path $plainPdf)) {
    throw "Slides PDF not found: $plainPdf"
}

Start-Process -FilePath $pympress -ArgumentList @("-t", "8", $plainPdf)
