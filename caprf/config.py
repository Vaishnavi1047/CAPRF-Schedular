"""All tunable settings in one place. Defaults follow Tables 1 and 2 of the CAPRF++ paper
(Felsy & Isaac Sajan, Real-Time Systems 2026); anything the paper leaves unspecified is
marked "impl." with the value we chose."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace


@dataclass
class ClusterConfig:  # Table 2
    n_edge: int = 150
    n_cloud: int = 50
    cloud_cores: tuple = (20, 32)
    cloud_mem_gb: tuple = (32, 64)
    edge_cores: tuple = (8, 16)
    edge_mem_gb: tuple = (8, 16)
    cloud_bw_mbps: tuple = (100, 1000)
    edge_bw_mbps: tuple = (10, 200)
    cloud_ghz: tuple = (2.4, 3.6)
    edge_ghz: tuple = (1.8, 2.5)
    cold_start_s: tuple = (0.05, 0.20)
    cloud_carbon: tuple = (0.3, 0.5)  # kg CO2 / kWh
    edge_carbon: tuple = (0.1, 0.3)
    # impl. (not given in the paper)
    cloud_prop_s: tuple = (0.040, 0.090)  # WAN round trip to the cloud
    edge_prop_s: tuple = (0.003, 0.012)
    cloud_watt_per_core: tuple = (9.0, 13.0)
    edge_watt_per_core: tuple = (3.5, 6.0)
    cold_start_watt: float = 25.0       # power drawn while a container initialises
    keepalive_s: float = 30.0           # idle container lifetime before it is reclaimed
    cloud_container_slots: int = 8      # warm runtimes a node can hold at once (LRU)
    edge_container_slots: int = 3
    carbon_diurnal_amp: float = 0.15    # +/- variation of grid carbon intensity over a day


@dataclass
class WorkloadConfig:
    dataset: str = "bitbrains"          # "bitbrains" | "kaggle"
    n_tasks: int = 50_000               # Table 1: 10k - 50k
    n_types: int = 8                    # workload classes found by k-means
    unseen_types: int = 2               # classes held out of training entirely (generalisation)
    train_frac: float = 0.7             # chronological train / test split
    target_load: float = 0.65           # mean cluster utilisation the arrival rate is scaled to
    bitbrains_vms: int = 300


@dataclass
class PPOConfig:  # Table 1
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    buffer_size: int = 50_000
    rollout: int = 2048
    hidden: int = 256
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    replay_frac: float = 0.5            # impl.: share of each update drawn from the FIFO buffer
    per_alpha: float = 0.6              # impl.: prioritised-replay exponent
    train_steps: int = 120_000


@dataclass
class CAPRFConfig:
    # WTC weights (Table 1): latency, energy, carbon, cold start
    alpha: float = 0.4
    beta: float = 0.3
    delta: float = 0.2
    rho: float = 0.1
    wtc_margin: float = 0.25            # Delta in Eq. (11), impl.
    # reward weights eta1..eta5 in Eq. (13), impl.
    eta: tuple = (1.0, 0.4, 0.3, 0.2, 0.1)
    # retrieval (Sec 3.2), impl.
    k_min: int = 16
    k_max: int = 64
    n_refs: int = 4                     # iDistance reference points
    btree_order: int = 16
    index_refresh: int = 64             # re-key all resources every N tasks
    avail_candidates: int = 8           # impl.: extra least-loaded nodes from the availability B+-tree
    sla_margin: float = 1.0             # impl.: WTC treats a node as SLA-feasible if L_ij <= margin*d_i
    # prewarming (Sec 3.5)
    prewarm_threshold: float = 0.35     # T_warm (Table 1)
    prewarm_top_k: int = 2
    arrival_window_s: float = 10.0      # "short future time window" for lambda_i, impl.
    embed_dim: int = 32
    type_embed_dim: int = 8
    unk_dropout: float = 0.1            # train the UNK type embedding used for unseen types


@dataclass
class Variant:
    """Feature switches; the ablation study (Tables 7/8) toggles these one by one."""
    name: str = "CAPRF++"
    context_encoding: bool = True
    btree_index: bool = True
    multi_objective: bool = True        # WTC filtering + multi-objective reward
    trajectory_buffer: bool = True
    prewarming: bool = True


ABLATION = [
    Variant("Baseline", False, False, False, False, False),
    Variant("+Context Encoding", True, False, False, False, False),
    Variant("+B+Tree Indexing", True, True, False, False, False),
    Variant("+Weighted MO-PPO", True, True, True, False, False),
    Variant("+Trajectory Buffer", True, True, True, True, False),
    Variant("CAPRF++ (full)", True, True, True, True, True),
]


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    caprf: CAPRFConfig = field(default_factory=CAPRFConfig)
    variant: Variant = field(default_factory=Variant)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
