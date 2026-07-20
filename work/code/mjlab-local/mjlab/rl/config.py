"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  distribution_cfg: dict[str, Any] | None = None
  """Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
  rnn_type: str | None = None
  """RNN type ("lstm" or "gru"). When set, class_name should be "RNNModel"."""
  rnn_hidden_dim: int = 256
  """Hidden state dimension for the RNN."""
  rnn_num_layers: int = 1
  """Number of stacked RNN layers."""
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel, CNNModel, or RNNModel)."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  share_cnn_encoders: bool = False
  """Share CNN encoders between actor and critic."""
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlReppoCriticCfg:
  """Configuration for REPPO's action-conditioned categorical critic."""

  hidden_dims: Tuple[int, ...] = (512, 512, 512)
  """Hidden dimensions of the Q network."""
  activation: str = "swish"
  """Hidden-layer activation."""
  obs_normalization: bool = True
  """Whether to normalize critic observations."""
  num_bins: int = 151
  """Number of HL-Gauss categorical bins."""
  vmin: float = -10.0
  """Lower endpoint of the categorical value support."""
  vmax: float = 10.0
  """Upper endpoint of the categorical value support."""
  sigma_ratio: float = 0.75
  """HL-Gauss standard deviation as a fraction of bin width."""
  zero_init_scale: float = 40.0
  """Scale of the trainable zero-return prior added to critic logits."""
  class_name: str = "ActionValueModel"
  """Critic class name resolved by RSL-RL."""


@dataclass
class RslRlReppoAlgorithmCfg:
  """Configuration for Relative Entropy Pathwise Policy Optimization."""

  num_learning_epochs: int = 8
  """Number of full critic and actor passes over each rollout."""
  num_mini_batches: int = 64
  """Number of mini-batches per epoch."""
  learning_rate: float = 3e-4
  """Learning rate shared by the actor and critic optimizers."""
  gamma: float = 0.99
  """Discount factor."""
  lam: float = 0.95
  """TD-lambda mixture for generalized Q targets."""
  max_grad_norm: float = 1.0
  """Maximum actor and critic gradient norms."""
  desired_kl: float = 0.01
  """Forward-KL constraint between behavior and current actors."""
  target_entropy: float = -0.5
  """Target differential entropy per action dimension."""
  init_alpha_temp: float = 0.1
  """Initial entropy-temperature multiplier."""
  init_alpha_kl: float = 0.1
  """Initial KL Lagrange multiplier."""
  num_kl_samples: int = 4
  """Number of behavior-policy samples used for Monte-Carlo KL estimation."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adamw"
  """Optimizer type."""
  weight_decay: float = 1e-3
  """AdamW weight decay used by the cvoelcker/rsl_rl implementation."""
  adam_betas: Tuple[float, float] = (0.9, 0.95)
  """Adam/AdamW beta coefficients."""
  rnd_cfg: dict[str, Any] | None = None
  """Reserved for runner compatibility; must remain None in this port."""
  symmetry_cfg: dict[str, Any] | None = None
  """Reserved for runner compatibility; must remain None in this port."""
  class_name: str = "REPPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlSacActorCfg:
  """Configuration for SAC's bounded tanh-Gaussian actor."""

  hidden_dims: Tuple[int, ...] = (256, 256, 256)
  """Hidden dimensions of the actor network."""
  activation: str = "elu"
  """Hidden-layer activation."""
  obs_normalization: bool = False
  """Whether to normalize actor observations."""
  init_noise_std: float = 1.0
  """Initial standard deviation of the action distribution."""
  layer_norm: bool = False
  """Whether to apply layer normalization in the actor network."""
  log_std_min: float = -20.0
  """Minimum log standard deviation."""
  log_std_max: float = 2.0
  """Maximum log standard deviation."""
  action_low: float = -1.0
  """Lower bound of the normalized action space."""
  action_high: float = 1.0
  """Upper bound of the normalized action space."""
  class_name: str = "SACActorModel"
  """Actor class name resolved by RSL-RL."""


@dataclass
class RslRlSacCriticCfg:
  """Configuration for SAC's twin action-value critic."""

  hidden_dims: Tuple[int, ...] = (256, 256, 256)
  """Hidden dimensions shared by both Q networks."""
  activation: str = "elu"
  """Hidden-layer activation."""
  obs_normalization: bool = False
  """Whether to normalize critic observations."""
  layer_norm: bool = False
  """Whether to apply layer normalization in the Q networks."""
  class_name: str = "SACCriticModel"
  """Critic class name resolved by RSL-RL."""


@dataclass
class RslRlSacAlgorithmCfg:
  """Configuration for Soft Actor-Critic."""

  replay_buffer_size: int = 1_000_000
  """Total replay capacity across all vectorized environments."""
  num_learning_epochs: int = 1
  """Number of passes through the requested replay updates."""
  num_mini_batches: int = 1
  """Number of replay mini-batches sampled per epoch."""
  mini_batch_size: int = 256
  """Number of replay transitions in each mini-batch."""
  actor_learning_rate: float = 3e-4
  """Actor optimizer learning rate."""
  critic_learning_rate: float = 3e-4
  """Critic optimizer learning rate."""
  alpha_learning_rate: float = 3e-4
  """Entropy-temperature optimizer learning rate."""
  actor_optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """Actor optimizer type."""
  critic_optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """Critic optimizer type."""
  auto_alpha: bool = True
  """Whether to learn the entropy temperature automatically."""
  alpha: float = 0.05
  """Initial entropy temperature."""
  tau: float = 0.005
  """Polyak target-network update coefficient."""
  gamma: float = 0.99
  """Discount factor."""
  target_entropy_scale: float = 1.0
  """Multiplier applied to the default target entropy."""
  max_grad_norm: float = 1.0
  """Maximum actor and critic gradient norm."""
  policy_frequency: int = 2
  """Number of critic updates between actor updates."""
  n_steps: int = 3
  """Number of replay steps used to construct return targets."""
  rnd_cfg: dict[str, Any] | None = None
  """Optional random-network-distillation configuration."""
  symmetry_cfg: dict[str, Any] | None = None
  """Optional symmetry augmentation configuration."""
  class_name: str = "SAC"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under ``{log_root}/{experiment_name}/``.
  The log root defaults to ``logs/rsl_rl`` and can be overridden with
  ``--log-root`` on the CLI."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlReppoRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration with REPPO-compatible actor and Q critic defaults."""

  class_name: str = "OnPolicyRunner"
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 512, 512),
      activation="swish",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "TanhGaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
        "std_range": (1e-4, 10.0),
        "action_range": (-1.0, 1.0),
      },
      class_name="LayerNormMLPModel",
    )
  )
  """Normalized stochastic actor configuration."""
  critic: RslRlReppoCriticCfg = field(default_factory=RslRlReppoCriticCfg)
  """Action-conditioned categorical critic configuration."""
  algorithm: RslRlReppoAlgorithmCfg = field(default_factory=RslRlReppoAlgorithmCfg)
  """REPPO algorithm configuration."""


@dataclass
class RslRlSacRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration for replay-based SAC training."""

  class_name: str = "OnPolicyRunner"
  """Mjlab uses its OnPolicyRunner adapter, which detects replay algorithms."""
  actor: RslRlSacActorCfg = field(default_factory=RslRlSacActorCfg)
  """Bounded stochastic actor configuration."""
  critic: RslRlSacCriticCfg = field(default_factory=RslRlSacCriticCfg)
  """Twin action-value critic configuration."""
  algorithm: RslRlSacAlgorithmCfg = field(default_factory=RslRlSacAlgorithmCfg)
  """SAC algorithm and replay configuration."""
  start_training: int = 0
  """Iteration at which replay updates may begin."""
