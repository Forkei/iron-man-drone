# M2.5 Task 4 — MJWarp State-Transfer Benchmark

Date: 2026-05-12  |  n_obstacles=4  |  steps=200  |  warmup=10  |  bench=50

## Latency breakdown (ms per batch_render call)

| N | Total (ms) | Transfer (ms) | Forward (ms) | Render+copy (ms) | Throughput (k steps/s) | Transfer% |
|---|---|---|---|---|---|---|
| 1 | 18.7 | 0.1 | 2.5 | 17.4 | 0.1 | 0.7% |
| 64 | 19.1 | 0.1 | 2.9 | 19.3 | 3.3 | 0.7% |
| 256 | 20.1 | 0.1 | 3.0 | 15.3 | 12.7 | 0.6% |
| 1024 | 30.6 | 0.2 | 2.8 | 18.7 | 33.5 | 0.6% |

## SC-4 gate

N=1024 throughput: **33.5k env-steps/sec** — PASS (gate: ≥ 25k)

## M3 architectural note

State-transfer overhead is 0.6% of total render time (≤ 20% threshold). Option A (MJX physics + MJWarp render) remains viable for M3 training.
