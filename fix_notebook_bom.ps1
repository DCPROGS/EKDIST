# fix_notebook_bom.ps1 — strip UTF-8 BOM from all .ipynb files
# Run from any directory: powershell -File E:\dcprogs\EKDIST\fix_notebook_bom.ps1

Get-ChildItem "E:\dcprogs\EKDIST" -Filter "*.ipynb" -Recurse |
    Where-Object { $_.DirectoryName -notmatch '\.ipynb_checkpoints' } |
    ForEach-Object {
        $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
        if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            [System.IO.File]::WriteAllBytes($_.FullName, $bytes[3..($bytes.Length-1)])
            Write-Host "Stripped BOM: $($_.Name)"
        }
    }
Write-Host "Done."
