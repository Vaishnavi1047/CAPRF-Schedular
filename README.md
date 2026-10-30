# CAPRF++ — working implementation

Implementation of **"CAPRF++: a context-aware B+ tree scheduler with weighted multi-objective
tradeoff guidance and predictive prewarming for generalized task scheduling in cloud systems"**
(C. Felsy & R. Isaac Sajan, *Real-Time Systems* 62:2, 2026, doi:10.1007/s11241-026-09456-0),
trained and evaluated on the paper's two datasets.

For every task arriving at a cloud-edge cluster, CAPRF++ chooses the machine that runs it. It
aims to meet the deadline (SLA) while keeping latency, energy, carbon and container cold-start
delay low, and to keep working on task types that were never seen during training.

## Quick start

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt -e .
#   (behind a corporate proxy: export UV_NATIVE_TLS=1 first)

.venv/bin/python scripts/download_data.py          # both datasets  -> data/raw/
.venv/bin/python -m caprf.experiments.run_all      # train + evaluate experiments
.venv/bin/python -m caprf.experiments.analyze      # tables, figures, statistics -> results/REPORT.md
.venv/bin/streamlit run app/dashboard.py           # presentation dashboard
.venv/bin/python -m pytest -q                      # unit tests (B+-tree, iDistance)
```
The final experiment configuration uses **5 seeds, 150,000 training steps and a 120-worker experiment pool**.

A quicker pass can be run with:

```bash
python -m caprf.experiments.run_all --seeds 1 --steps 40000
```


## Datasets

| | Dataset 1 — GWA-T-12 Bitbrains | Dataset 2 — Kaggle Cloud Task Scheduling |
|---|---|---|
| Source | Official host (gwa.ewi.tudelft.nl) is offline; raw fastStorage CSVs pulled unmodified from the public mirror `muse-research-lab/cloud-forecast-data-persistence` | Kaggle `ziya07/cloud-task-scheduling-dataset` needs a login; an identical public copy is used (same 20,000 rows and columns). With Kaggle credentials: `kaggle datasets download ziya07/cloud-task-scheduling-dataset -p data/raw/kaggle --unzip` |
| Content | 1,250 VMs, 5-minute CPU/memory/disk/network samples, 30 days (300 VMs used by default; `--vms 1250` for all) | 20,000 tasks: CPU %, RAM, disk IO, network IO, priority, VM, execution time, SLA target |
| Tasks used | 50,000 active 5-minute samples | all 20,000 |
| Arrivals | real trace timestamps, 30 days compressed into the simulation horizon | no timestamps in the data → Markov-modulated Poisson process |
| Task types | 8 k-means classes of VM usage profiles, 2 held out as unseen | 6 k-means classes of task profiles, 1 held out as unseen |

Observed while profiling the Kaggle data: its features are uniform and mutually uncorrelated, i.e. it
is synthetic, and its `Target` column is essentially a threshold on `Execution_Time` (correlation −0.86).

**Splits.** Chronological 70/30 train/test. Unseen task types are removed from training entirely and
appear only in the test window. The last 15% of the training period is a validation slice used to pick
the best checkpoint; the test window is never used for training or selection.

## How the paper maps to the code

| Paper | Code |
|---|---|
| §3.1 Eq. 1 z-score over a moving window | `caprf/models/networks.py` · `RunningNorm` |
| §3.1 Eq. 2–6 type embedding + two-layer encoder | `caprf/models/networks.py` · `TaskEncoder` |
| §3.2 Eq. 7 resource encoder ψ(a, e, l, γ) | `caprf/models/networks.py` · `ResourceEncoder` |
| §3.2 B+-tree index, Eq. 8–9 distance threshold τ, empty-set fallback | `caprf/index/bplustree.py` · `BPlusTree`, `IDistanceIndex`, `AdaptiveRetriever` |
| §3.3 Eq. 10–11 WTC score Ω and feasible set C_i | `caprf/schedulers/caprf.py` · `_tradeoff`, `_feasible` |
| §3.3 latency/energy/carbon estimates L ∝ c/κ, E = P·L, C = γ·E | `caprf/sim/cluster.py` · `Cluster.predict` |
| §3.4 Eq. 12 state, Eq. 13 multi-objective reward | `caprf/schedulers/caprf.py` · `decide`, `observe` |
| §3.4 Eq. 14–15 clipped PPO + GAE (Algorithm 2) | `caprf/agent/ppo.py` · `PPOLearner`, `gae` |
| §3.4 FIFO trajectory buffer + prioritised replay (Fig. 2) | `caprf/agent/ppo.py` · `Storage`, `PPOLearner.update` |
| §3.5 Eq. 16 arrival predictor λ, Eq. 17–18 prewarming rule | `ArrivalHead` + prewarming step in `CAPRFScheduler.decide` |
| Algorithm 1 (actor) / Algorithm 3 (pipeline) | `caprf/schedulers/caprf.py` · `CAPRFScheduler` |
| §4.1 custom cloud-edge simulator, Table 2 | `caprf/sim/cluster.py` |
| Table 1 hyper-parameters | `caprf/config.py` |
| §4 experiments, Tables 3–9, Figs. 3–8 | `caprf/experiments/run_all.py`, `analyze.py` |

## Implementation choices the paper leaves open

- **B+-tree nearest-neighbour search.** A B+-tree orders one-dimensional keys only, so Euclidean
  retrieval (Eq. 9) uses **iDistance** (Jagadish et al., 2005): key = partition·C + distance to a
  reference point, and a ball query becomes a few B+-tree range scans followed by exact checks.
  A second B+-tree, keyed on each node's queue backlog, adds the 8 least-loaded nodes to the candidate
  set. This follows the paper's "live availability" ranking and was needed for good candidates under
  bursty load.
- **Implicit metric learning.** The actor's logit includes −β‖z_t − z_r‖, so PPO gradients pull
  rewarded task/resource pairs together (§3.2) without an explicit contrastive loss.
- **Unseen task types** map to a learned UNK embedding, trained by replacing 10% of type ids with UNK.
- **Sustainability preference s_i** scales the carbon term of the reward by (0.5 + s_i).
- **Prewarming** is evaluated on the top-2 policy-ranked machines other than the chosen one (the chosen
  machine gets a container anyway). Prewarming energy is charged to the task's reward and to all reported totals.
- **Training.** Episodes of 4,000 tasks are drawn from the training period at a random load in
  [target − 0.15, target + 0.10] (domain randomisation). The checkpoint with the best validation reward is kept.
- **Load.** Arrival rates are scaled so mean utilisation is 0.72 (Bitbrains) or 0.90 (Kaggle).
  Kaggle arrivals are smoother, so it needs higher load to stress the cluster.
- **Baselines.** The paper's six baselines have no public code. We compare against reproducible
  schedulers covering the same design space: Random, Round-Robin, Least-Loaded, Greedy-MCT (online
  Min-Min), Green-Greedy, WTC-Greedy (Eq. 19 without learning), and vanilla PPO (the ablation baseline).
- **Reductions** (carbon, cold start) are relative to **Least-Loaded**, a classic load balancer with no
  sustainability objective and no prewarming (the paper's "traditional allocation", §4.3.1).
  Round-Robin is unsuitable as the reference: it emits little carbon only because it misses about 60% of deadlines.
- **Generalization (%)** = 100 × SLA on unseen-type tasks / SLA on seen-type tasks in the test window, capped at 100.

## Seeds: what is random and what is fixed

The final experiment configuration uses **5 random seeds (0–4)**. A seed changes stochastic aspects of training and simulation while preserving the underlying scheduling problem.

**Varies with the seed for CAPRF++ and other learned variants:**

- initial neural-network weights (`torch.manual_seed(seed)`);
- which 4,000-task windows of the training period are drawn as episodes, and the random load of each
  episode (domain randomisation);
- exploration: actions are *sampled* from the policy during training, plus 4 random extra candidate
  machines per decision;
- which type ids are replaced by UNK during training (10%), and which buffer samples are replayed (PER);
- the reference points of the iDistance index;
- the runtime noise of the simulator at test time (log-normal ±10% on execution and cold-start times).

**Fixed across seeds:**
- the workload (the same tasks, arrivals, deadlines and train/test split);
- which task types are held out as unseen;
- the cluster hardware (the same 200 machines and their parameters);
- all hyper-parameters.

**Baselines** (Round-Robin, Greedy-MCT, …) are deterministic rules, so only the simulator's runtime noise changes between their seeds. CAPRF++ and the other learned variants additionally include training-to-training variation arising from stochastic initialization, exploration and training.

For a given seed, CAPRF++ and every baseline see the same runtime noise, so their results can be compared pair by pair (this pairing is what the Wilcoxon tests use).


## Results (5 seeds)

Full tables and figures: `results/REPORT.md`, `results/tables/`, `results/figures/`.

These are the results after the full training runs (150,000 training steps, 5 seeds). For each seed, the checkpoint with the best validation reward is evaluated on the untouched test split. CAPRF++ reaches 85.7% SLA on Bitbrains and 98.5% on Kaggle, with cold-start reductions of 94.9% and 91.0% respectively. The corresponding unseen-workload generalization is 100.0% and 99.2%.

| Dataset | Scheduler | SLA % | Latency (ms) | Carbon reduction % | Cold-start reduction % | Generalization % |
|---|---|---|---|---|---|---|
| Bitbrains | **CAPRF++** | 85.7 | 2,966 | 12.9 | 94.9 | 100.0 |
| Bitbrains | Least-Loaded | 92.9 | 1,927 | 0.0 | 0.0 | 98.7 |
| Bitbrains | Greedy-MCT | 92.8 | 1,874 | -14.6 | 79.4 | 98.6 |
| Bitbrains | WTC-Greedy | 83.5 | 4,439 | 20.7 | 83.9 | 100.0 |
| Kaggle | **CAPRF++** | 98.5 | 2,257 | 9.1 | 91.0 | 99.2 |
| Kaggle | Least-Loaded | 99.5 | 1,518 | 0.0 | 0.0 | 99.8 |
| Kaggle | Greedy-MCT | 100.0 | 1,525 | -2.7 | 57.0 | 100.0 |
| Kaggle | WTC-Greedy | 92.3 | 3,740 | 20.7 | 67.4 | 100.0 |


**What holds up from the paper's claims.**
- Predictive prewarming cuts cold-start delay by about 90%, the most of any scheduler.
- Generalization to unseen task types is 99–100%.
- Carbon is about 9–13% lower than Least-Loaded, and CAPRF++ also uses less carbon than Greedy-MCT on both datasets.
- A scheduling decision takes about 2.88 ms on Bitbrains and 2.95 ms on Kaggle, with p99 values of about 5.11 ms and 5.41 ms respectively.


**What does not.** CAPRF++ does not beat every baseline on every metric at once. It sits in the middle of the SLA-versus-carbon trade-off: about 7 SLA points (Bitbrains) or 1–2 points (Kaggle) below the latency-greedy schedulers, with clearly better SLA and latency than the carbon-greedy ones. Paired interval-level Wilcoxon tests against Greedy-MCT show significant differences for carbon (p = 0.0186 on Bitbrains and p = 0.0098 on Kaggle) and cold-start overhead (p = 0.0049 and p = 0.0010 respectively); the statistical claims are limited to these interval-level comparisons.


**B+-tree.** It is correct (unit-tested, O(log N) updates), but with the learned embeddings the task queries lie far from the resource manifold, so iDistance prunes little and is slower than a vectorised NumPy scan (`results/figures/fig_index_benchmark.png`). Candidate quality comes mainly from the availability-keyed B+-tree.


## Dashboard

`app/dashboard.py` provides a Streamlit dashboard over the same workloads, trained checkpoints,
simulation code and generated experiment results used above.

```bash
.venv/bin/streamlit run app/dashboard.py
```

The dashboard contains three main views:

- **Live Interactive Demo** — runs CAPRF++ and the baseline schedulers on test tasks and compares their scheduling behaviour and performance.
- **Explainability & Walkthrough** — exposes the scheduling pipeline, candidate selection, WTC filtering, PPO decisions and prewarming behaviour.
- **Official Experimental Results** — presents the generated results, baseline comparisons, ablations, training curves, prewarming, scalability, statistical tests and B+-tree benchmark from `results/`.

The dashboard reads trained checkpoints and analysis outputs from `results/`, so the displayed results stay tied to the experiment pipeline rather than a separate demo implementation.


## Layout

```
caprf/config.py              all hyper-parameters (Tables 1-2) and ablation variants
caprf/data/workload.py       dataset -> task streams, splits, load calibration
caprf/sim/cluster.py         cloud-edge simulator (queues, containers, energy, carbon)
caprf/sim/runner.py          episode runner + metrics
caprf/index/bplustree.py     B+-tree, iDistance index, adaptive retriever
caprf/models/networks.py     encoders, actor-critic, arrival predictor
caprf/agent/ppo.py           PPO learner, FIFO trajectory buffer, PER
caprf/schedulers/caprf.py    CAPRF++ scheduler (Algorithms 1 and 3)
caprf/schedulers/heuristics.py  baseline schedulers
caprf/experiments/           run_all.py (experiments), analyze.py (report), common.py
app/dashboard.py             Streamlit dashboard
scripts/download_data.py     dataset download
results/                     REPORT.md, tables/, figures/, runs/ (models, logs, per-task CSVs)
```