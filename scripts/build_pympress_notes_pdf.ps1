$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $projectRoot
try {
    xelatex -disable-installer -interaction=nonstopmode -halt-on-error gnn_bandgap_research_final_slides.tex
    python .\scripts\build_pympress_sidecar_pdf.py
} finally {
    Pop-Location
}
