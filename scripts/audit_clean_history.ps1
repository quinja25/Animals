param(
    [string]$GitDir = ".git"
)

$ErrorActionPreference = "Stop"

$patterns = @(
    '(^|/)submissions/',
    '(^|/)scratch/',
    '\.DS_Store$',
    '__pycache__/',
    '\.pyc$',
    '(^|/)data/',
    'hanwoo_train\.csv$',
    'test_hanwoo\.csv$',
    'hanwoo_lineage(_0612)?\.csv$',
    'hanwoo_weather\.csv$',
    'hanwoo_area\.csv$',
    'hanwoo_death\.csv$'
)

$objects = git --git-dir=$GitDir rev-list --objects --all
$matches = foreach ($line in $objects) {
    foreach ($pattern in $patterns) {
        if ($line -match $pattern) {
            $line
            break
        }
    }
}

if ($matches) {
    Write-Error ("Sensitive or generated paths remain in history:`n" + ($matches -join "`n"))
}

git --git-dir=$GitDir fsck --no-progress | Out-Host
Write-Output "Clean-history audit passed for $GitDir"
