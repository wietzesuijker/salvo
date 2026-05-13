# Security policy

If you find a vulnerability in salvo, please report it privately rather than opening a public issue. Open a GitHub security advisory on this repo, or email the maintainer listed in `pyproject.toml`. Expect an acknowledgement within 7 days.

salvo dispatches user code to SLURM clusters via `sbatch`. It does not transmit credentials, secrets, or job payloads to any third party. The only network calls are SSH/scp/rsync against clusters the operator configures.
