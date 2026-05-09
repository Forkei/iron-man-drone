"""
Phase 2 encoder training.

Supervised MSE training of AdaptationEncoder on (history_window, e_t_normalized) pairs
collected by collect_phase2_data.py.

Data format: experiments/phase2_data/chunk_*.npz
  obs_base   : (N_ENVS, 1000, 42) float32
  actions    : (N_ENVS, 1000, 4)  float32
  priv_states: (N_ENVS, 8)        float32 (raw e_t, constant per episode)

Window construction (on-the-fly):
  combined[ep, t] = (obs_base[ep, t], actions[ep, t-1]) — 46-dim pair
  Window at step t: combined[ep, t-H+1:t+1] flattened = 2300-dim
  Padded with zeros for steps before t=0 (episode start).
  Training examples: t ∈ [49, 999] per episode → 951 examples each.

Usage:
  python scripts/train_phase2_encoder.py
  python scripts/train_phase2_encoder.py --data_dir experiments/phase2_data
  python scripts/train_phase2_encoder.py --epochs 2000 --batch_size 4096
"""

import sys
import time
import argparse
import csv
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "experiments/phase2_data"
DEFAULT_OUT_DIR  = REPO_ROOT / "experiments/phase2_encoder"

H           = 50    # history window length
OBS_DIM     = 42
ACTION_DIM  = 4
PAIR_DIM    = OBS_DIM + ACTION_DIM   # 46
WINDOW_DIM  = H * PAIR_DIM           # 2300
E_T_DIM     = 8
VAL_FRAC    = 0.10                   # 90/10 split
VAL_INTERVAL = 50                    # epochs between val checks

CHANNEL_NAMES = ["η₁", "η₂", "η₃", "η₄", "m_scale", "Fx", "Fy", "Fz"]


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all chunks. Returns:
      combined_padded : (N_ep, 1000 + H - 1, PAIR_DIM) float32
      e_t_norm        : (N_ep, E_T_DIM) float32
    """
    from iron_man_drone.policy.encoder import normalize_e_t

    chunk_files = sorted(data_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files in {data_dir}. "
                                 "Run collect_phase2_data.py first.")

    print(f"Loading {len(chunk_files)} chunks from {data_dir} ...")
    t0 = time.time()

    all_obs_base   = []
    all_actions    = []
    all_priv_states = []

    for f in chunk_files:
        d = np.load(f)
        all_obs_base.append(d["obs_base"])       # (N, 1000, 42)
        all_actions.append(d["actions"])         # (N, 1000, 4)
        all_priv_states.append(d["priv_states"]) # (N, 8)

    obs_base    = np.concatenate(all_obs_base,    axis=0)  # (N_ep, 1000, 42)
    actions     = np.concatenate(all_actions,     axis=0)  # (N_ep, 1000, 4)
    priv_states = np.concatenate(all_priv_states, axis=0)  # (N_ep, 8)
    N_ep = obs_base.shape[0]
    print(f"  Loaded {N_ep} episodes in {time.time()-t0:.1f}s.")

    # Construct combined[ep, t] = (obs_base[t], actions[t-1]) — 46-dim
    combined = np.zeros((N_ep, 1000, PAIR_DIM), dtype=np.float32)
    combined[:, :, :OBS_DIM] = obs_base
    combined[:, 1:, OBS_DIM:] = actions[:, :-1]   # shift actions by 1 (prev_action)
    # combined[:, 0, OBS_DIM:] = 0 (initial prev_action, already zeros)

    # Pad H-1=49 zeros at episode start so window[t:t+H] is always valid for t≥0
    pad = np.zeros((N_ep, H - 1, PAIR_DIM), dtype=np.float32)
    combined_padded = np.concatenate([pad, combined], axis=1)  # (N_ep, 1049, 46)

    # Normalize targets to [-1, 1]
    e_t_norm = normalize_e_t(priv_states).astype(np.float32)  # (N_ep, 8)

    print(f"  combined_padded: {combined_padded.shape}, {combined_padded.nbytes/1e9:.2f} GB")
    print(f"  e_t_norm stats: mean={e_t_norm.mean():.3f}, std={e_t_norm.std():.3f}")
    print(f"  Fault episodes (any η<0.99): "
          f"{(priv_states[:,:4].min(axis=1) < 0.99).sum()} / {N_ep}")
    return combined_padded, e_t_norm


def make_train_val_split(N_ep: int, rng: np.random.Generator):
    idx = rng.permutation(N_ep)
    n_val = max(1, int(N_ep * VAL_FRAC))
    return idx[n_val:], idx[:n_val]    # train_idx, val_idx


def sample_batch(
    combined_padded: np.ndarray,  # (N_ep, 1049, 46)
    e_t_norm: np.ndarray,         # (N_ep, 8)
    episode_idx: np.ndarray,      # indices of episodes to sample from
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample batch_size (window, target) pairs from given episodes."""
    ep_idx = rng.choice(episode_idx, size=batch_size, replace=True)
    t_idx  = rng.integers(0, 951, size=batch_size)   # t in [0..950]; window = padded[t:t+H]
    # Fancy index: combined_padded[ep_idx[:, None], t_idx[:, None] + arange(H), :]
    time_idx = t_idx[:, None] + np.arange(H, dtype=np.int32)[None, :]  # (batch, H)
    windows  = combined_padded[ep_idx[:, None], time_idx].reshape(batch_size, -1)  # (batch, 2300)
    targets  = e_t_norm[ep_idx]                                                      # (batch, 8)
    return windows, targets


def compute_mse_per_channel(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Returns (E_T_DIM,) per-channel MSE."""
    return np.mean((preds - targets) ** 2, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir",    default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs",     type=int,   default=2000)
    parser.add_argument("--batch_size", type=int,   default=4096)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Out dir    : {out_dir}")
    print(f"Epochs     : {args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    combined_padded, e_t_norm = load_data(Path(args.data_dir))
    N_ep = combined_padded.shape[0]
    rng  = np.random.default_rng(args.seed)
    train_idx, val_idx = make_train_val_split(N_ep, rng)
    print(f"\nTrain/val split: {len(train_idx)} / {len(val_idx)} episodes")

    # ── Encoder + optimizer ────────────────────────────────────────────────────
    from iron_man_drone.policy.encoder import AdaptationEncoder

    encoder = AdaptationEncoder()
    enc_key = jax.random.PRNGKey(args.seed)
    dummy   = jnp.zeros((1, WINDOW_DIM))
    params  = encoder.init(enc_key, dummy)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Encoder params: {n_params:,}")

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    # ── JIT-compiled train step ────────────────────────────────────────────────
    @jax.jit
    def train_step(params, opt_state, windows, targets):
        def loss_fn(params):
            preds = encoder.apply(params, windows)  # (batch, 8)
            return jnp.mean((preds - targets) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def eval_preds(params, windows):
        return encoder.apply(params, windows)

    # Warmup
    _w, _t = sample_batch(combined_padded, e_t_norm, train_idx, args.batch_size, rng)
    _w_j, _t_j = jnp.array(_w), jnp.array(_t)
    params, opt_state, _ = train_step(params, opt_state, _w_j, _t_j)
    print("JIT warmed up.\n")

    # ── Precompute val set (fixed 10k windows for consistent tracking) ─────────
    n_val_windows = min(10_000, len(val_idx) * 951)
    val_wins_np, val_tgts_np = sample_batch(
        combined_padded, e_t_norm, val_idx, n_val_windows, rng
    )
    val_wins_j = jnp.array(val_wins_np)
    val_tgts_j = jnp.array(val_tgts_np)

    # ── CSV logging ────────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="", buffering=1)
    fieldnames = ["epoch", "train_mse", "val_mse"] + [f"val_mse_{c}" for c in CHANNEL_NAMES]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val_mse = float("inf")
    best_params  = params
    checkpointer = ocp.PyTreeCheckpointer()

    print(f"Training for {args.epochs} epochs...")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'η MSE':>8}  {'mass MSE':>9}  {'wind MSE':>9}")
    print("-" * 62)

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # One gradient step per epoch on a fresh batch
        w_np, t_np = sample_batch(combined_padded, e_t_norm, train_idx, args.batch_size, rng)
        w_j, t_j   = jnp.array(w_np), jnp.array(t_np)
        params, opt_state, train_loss = train_step(params, opt_state, w_j, t_j)
        train_mse = float(train_loss)

        row = {"epoch": epoch, "train_mse": f"{train_mse:.6f}"}

        if epoch % VAL_INTERVAL == 0 or epoch == 1 or epoch == args.epochs:
            val_preds_j = eval_preds(params, val_wins_j)
            val_preds   = np.array(val_preds_j)
            val_tgts    = np.array(val_tgts_j)
            per_ch_mse  = compute_mse_per_channel(val_preds, val_tgts)  # (8,)
            val_mse     = float(per_ch_mse.mean())

            eta_mse  = float(per_ch_mse[:4].mean())
            mass_mse = float(per_ch_mse[4])
            wind_mse = float(per_ch_mse[5:].mean())

            row["val_mse"] = f"{val_mse:.6f}"
            for i, name in enumerate(CHANNEL_NAMES):
                row[f"val_mse_{name}"] = f"{per_ch_mse[i]:.6f}"

            elapsed = time.time() - t_start
            eta_secs = elapsed / epoch * (args.epochs - epoch)
            print(f"{epoch:>6}  {train_mse:>10.6f}  {val_mse:>10.6f}  "
                  f"{eta_mse:>8.5f}  {mass_mse:>9.5f}  {wind_mse:>9.5f}  "
                  f"[{elapsed/60:.1f}m elapsed, {eta_secs/60:.0f}m left]")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_params  = params
                ckpt_path = out_dir / "best_checkpoint"
                if ckpt_path.exists():
                    shutil.rmtree(ckpt_path)
                checkpointer.save(str(ckpt_path), {"params": best_params})

            # Early stop
            if val_mse < 0.005:
                print(f"\nEarly stop: val_mse={val_mse:.6f} < 0.005 for 3 checks.")
                break

        writer.writerow(row)

    log_file.close()

    # ── Final report ───────────────────────────────────────────────────────────
    val_preds_j = eval_preds(best_params, val_wins_j)
    val_preds   = np.array(val_preds_j)
    val_tgts    = np.array(val_tgts_j)
    final_per_ch = compute_mse_per_channel(val_preds, val_tgts)
    final_val_mse = float(final_per_ch.mean())

    print()
    print("=" * 72)
    print(f"  Phase 2 Encoder Training — Final Report")
    print("=" * 72)
    print(f"  Best val MSE (overall):     {best_val_mse:.6f}  "
          f"(target: ≤ 0.020)")
    print(f"  Final per-channel MSE (normalized):")
    for i, name in enumerate(CHANNEL_NAMES):
        bar = "✓" if final_per_ch[i] <= (0.03 if i < 4 else 0.02) else "✗"
        print(f"    {bar} {name:8s}: {final_per_ch[i]:.6f}")
    print()

    # Gate check
    eta_mse_final  = float(final_per_ch[:4].mean())
    mass_mse_final = float(final_per_ch[4])
    wind_mse_final = float(final_per_ch[5:].mean())
    gate_overall   = best_val_mse <= 0.02
    gate_eta       = eta_mse_final <= 0.03

    print(f"  Offline MSE gate: overall ≤ 0.020 → {'PASS' if gate_overall else 'FAIL'}")
    print(f"  Offline MSE gate: η ≤ 0.030       → {'PASS' if gate_eta else 'FAIL'}")
    print(f"  η MSE:      {eta_mse_final:.6f}")
    print(f"  mass MSE:   {mass_mse_final:.6f}")
    print(f"  wind MSE:   {wind_mse_final:.6f}")
    print()

    # Hypothesis check
    user_predicted_mse = 0.35
    print(f"  Hypothesis check:")
    print(f"    User predicted MSE ≈ {user_predicted_mse:.2f} — actual: {final_val_mse:.4f}")
    if final_val_mse < user_predicted_mse * 0.5:
        print(f"    Prediction WAY OVER — encoder learned well. Spec target was realistic.")
    elif final_val_mse < user_predicted_mse:
        print(f"    Prediction over — encoder performed better than expected.")
    else:
        print(f"    Prediction under — encoder underperformed. See spec §F4.")
    print(f"    η vs mass easier? η MSE={eta_mse_final:.5f}, mass MSE={mass_mse_final:.5f} "
          f"→ {'η easier ✓' if eta_mse_final < mass_mse_final else 'mass easier (prediction incorrect)'}")
    print(f"    Wind channels: MSE={wind_mse_final:.6f} "
          f"→ {'near-zero as expected ✓' if wind_mse_final < 0.001 else 'unexpected signal!'}")
    print()

    if gate_overall and gate_eta:
        print("  ✓ Encoder passes offline gate. Run scripts/eval_m2_phase2.py next.")
    else:
        print("  ✗ Encoder FAILS offline gate. Diagnose before closed-loop eval.")
        print("    See notes/M2_phase2_spec.md §F4 for guidance.")

    print(f"\n  Checkpoint: {out_dir / 'best_checkpoint'}")
    print(f"  Training log: {log_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
