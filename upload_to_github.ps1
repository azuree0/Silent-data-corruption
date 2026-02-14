# Upload to GitHub - run after installing Git for Windows
# https://git-scm.com/download/win

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

git init
git add .
git commit -m "Initial commit: SDC test tool (CPU, RAM, SSD, GPU)"
git remote add origin https://github.com/azuree0/Silent-data-corruption.git
git branch -M main
git push -u origin main
