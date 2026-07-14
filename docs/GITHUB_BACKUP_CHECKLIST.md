# GitHub Backup Checklist

## Commit Scope

This backup commit includes core QFi-CR / CDPruner code, selected evaluation loader changes, selected experiment scripts, `.gitignore`, `README.md` when staged, and documentation under `docs/`.

## Excluded Paths

The backup explicitly excludes local secrets, generated outputs, logs, checkpoints, datasets, playground data, model weights, JSONL answer/debug files, and other large binary artifacts.

Excluded examples:

- `secrets/`
- `secrets/api.yaml`
- `.env`
- `outputs/`
- `logs/`
- `checkpoints/`
- `datasets/`
- `playground/data/`
- `*.jsonl`
- `*.pt`
- `*.pth`
- `*.safetensors`

## Remote

- Remote name: `qfi-prune`
- Remote URL: `https://github.com/xiaorong-ui/qfi-prune.git`
- The GitHub token is not stored in the remote URL.

## Push Branch

- Branch: `qficr-backup`
- No force push is used.

## Commit Message

`Backup QFi-CR core code and experiment scripts`

## Sensitive Information Check

Before committing, staged files are checked for forbidden paths and common secret patterns. Any token-like matches must be reviewed before committing.

## Restore Notes

To restore or inspect this backup from a clean machine:

```bash
git clone https://github.com/xiaorong-ui/qfi-prune.git
cd qfi-prune
git checkout qficr-backup
```

Install dependencies and recreate local configuration separately. Runtime outputs, datasets, checkpoints, model weights, and secrets are intentionally not part of this backup.

## Secrets

`secrets/api.yaml` is not committed. Recreate it locally when needed from a private source.
