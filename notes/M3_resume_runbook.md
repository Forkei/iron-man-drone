# M3 Training Resume — Runbook (tonight)

Resume `m3_run1` from epoch 5899 (~193M steps, 38%) to completion (500M / epoch 15259).

## The command

```bash
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/forke/Documents/Drone/iron-man-drone && /home/forke/jax_env/bin/python -u scripts/train_m3.py --run_name m3_run1 2>&1 | tee -a m3_run1_resume.log"
```

- `--run_name m3_run1` is **required** — without it the script starts a fresh
  timestamped run instead of resuming. With it, the script auto-detects the
  latest checkpoint in `/home/forke/m3_checkpoints/m3_run1/` (epoch_005899) and
  continues from epoch 5900. The CSV log appends; nothing is overwritten.
- The interactive gates are already passed (H1 @916, 1-hour pause @3576), and the
  resume starts at epoch 5900 > both, so **no prompts will block** — safe to leave.

## Before you start (recommended)

Stop the periodic killer so the run isn't interrupted every ~3–4h
(the resume system survives kills, but this avoids the churn):

```bash
wsl -d Ubuntu -- sudo systemctl mask unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
```

## What to expect

- Throughput is render-bound; the remaining ~9,360 epochs take roughly one long
  evening / overnight. Checkpoints save every 50 epochs (~minutes apart) to
  `/home/forke/m3_checkpoints/m3_run1/`.
- If interrupted (SIGTERM / sleep), just re-run the exact same command — it
  resumes from the last checkpoint (max ~5 min lost).
- NaN guard is active (aborts + checkpoints if weights blow up).

## When it finishes (epoch 15259 / "Training complete")

Final checkpoint: `/home/forke/m3_checkpoints/m3_run1/final`

Re-run the eval to compare against the 38% baseline (this session's numbers):

```bash
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/forke/Documents/Drone/iron-man-drone && /home/forke/jax_env/bin/python -u scripts/eval_m3.py --ckpt /home/forke/m3_checkpoints/m3_run1/final --n 128 --out /tmp/m3_eval_final.json"
```

## Baseline to beat (epoch 5899, this session — see M3_eval_baseline.md)

- random nominal: fair CF ~91.5%, unfair-bucket CF (save-rate) ~24%
- fault70: fair CF ~61%, OOB ~33%
- tracking MED ~0.15m (target ≤0.10m)

Goal at 500M: tracking MED down toward 0.07–0.10m, fault OOB down, unfair-bucket
save-rate up (proving the deviate-to-survive skill got more reliable, not just
that tracking sharpened).
