# M1 Hypothesis

## What I'm testing
First training run of SimpleFlight recipe in MJX. Goal: confirm the
recipe converges on this stack before any deviations.

## What I expect to see in the first 30 minutes
- Reward: starts around 0.4 on average (check sanity_check output), trends up
- Value loss: finite, decreasing
- Policy entropy: starts around idk the numbers but quite high, very slowly decreasing (NOT collapsing to 0)
- Episode length: starts short (drone falls), increases as policy learns. I guess like basically 0.4s?

## What I expect at convergence (~4-5h)
- Eval MED on figure-eight (normal): < 0.06m, yeah I think we can get there. Probably 0.08-0.1
- Eval MED on figure-eight (slow): < 0.05m, probably a little less, I'm not super confident we can get this first try without tuning things, never does
- Eval MED on figure-eight (fast): < 0.15m, sure, still sounds high, might get there sometimes but I think it'll either be like that or have found an exploit to go wide and back, not smooth
- Smooth velocity profiles, no oscillation, we're hoping, chance that it doesn't

## Failure signals (and what each means)
- Reward flatlines at random-policy value → obs/reward broken, NOT a hyperparam issue
- Reward goes up then crashes → entropy collapse or value function divergence
- Entropy hits zero in <500 epochs → entropy_coef too low or reward magnitudes wrong
- Value loss explodes → critic LR too high, or actor-critic accidentally sharing weights
- Episode length stays at minimum → drone can't even hover, check CTBR mixer, I feel this'll be kind of probable

## Time budget
12h hard cap. Check at 1h, 4h, 8h.

## What I do at the 1h checkpoint
If reward isn't trending up by 1h, STOP. Re-read paper Section IV-B
and the spec, do not tweak hyperparameters first.

## What I do on success
Archive checkpoint as m1-baseline. Write M1_results.md comparing
numbers to paper. Propose M2 brief.

## What I do on failure
Re-read SimpleFlight Sections III-D and IV-B. Diagnose per the
debugging order in the spec. ONE fix at a time, not a batch.
