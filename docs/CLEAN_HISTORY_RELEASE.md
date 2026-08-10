# Private Clean-History Release Checklist

Use this checklist before recreating `quinja25/livestock-quality-prediction` and making it public.

## Stop Conditions

- Do not delete the existing GitHub repository until a local private mirror backup exists and passes `git fsck`.
- Do not push a public repository until the cleaned mirror audit returns no sensitive paths from every ref.
- Do not publish the LinkedIn URL until a fresh outsider clone of the recreated repository passes the same audit.

## Teammate Notice

Send this before rewriting or recreating the GitHub repository:

```text
I am cleaning and recreating the livestock-quality-prediction repository to remove private competition artifacts from Git history. Commit SHAs will change. Please stop pushing to the old repository and re-clone the recreated repository after I confirm the clean-history audit passed.
```

## Local Backup

Create and verify a private mirror backup first:

```powershell
New-Item -ItemType Directory -Force -Path backups | Out-Null
git clone --mirror https://github.com/quinja25/livestock-quality-prediction.git backups/livestock-quality-prediction-private-backup.git
git --git-dir=backups/livestock-quality-prediction-private-backup.git fsck --no-progress
```

Keep this backup private. It may still contain the original sensitive history.

## Clean Mirror

Create a separate mirror and rewrite only that copy:

```powershell
git clone --mirror backups/livestock-quality-prediction-private-backup.git backups/livestock-quality-prediction-cleaned.git
Set-Location backups/livestock-quality-prediction-cleaned.git
python -m git_filter_repo --path submissions/ --path scratch/ --path .DS_Store --path-glob "*/__pycache__/*" --invert-paths --force
Set-Location ../..
```

## Audit

Run the helper against the cleaned mirror:

```powershell
.\scripts\audit_clean_history.ps1 -GitDir backups/livestock-quality-prediction-cleaned.git
```

The audit must report no matches for:

- `submissions/`
- `scratch/`
- `.DS_Store`
- committed `__pycache__` files
- common local competition data filenames

## GitHub Recreation

After the local backup and cleaned mirror pass:

1. Make the current GitHub repository private.
2. Delete the old GitHub repository.
3. Recreate `quinja25/livestock-quality-prediction` as private.
4. Push the cleaned mirror to the recreated private repository.
5. Commit and push the current README, notebook, license, images, and pipeline improvements onto the cleaned private repository.
6. Clone the recreated repository into a fresh directory.
7. Run `scripts/audit_clean_history.ps1` from the fresh clone or against its `.git`.
8. Inspect the repository as an outsider: README renders, images load, notebook opens, license is present, and no private data is present.
9. Only after the fresh clone audit passes, switch the repository to public.

Keep the repository name unchanged so the eventual URL remains:

```text
GitHub - quinja25/livestock-quality-prediction · GitHub
```
