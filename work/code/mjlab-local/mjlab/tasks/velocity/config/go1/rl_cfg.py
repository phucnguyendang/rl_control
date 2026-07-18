"""RL configuration for Unitree Go1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlReppoAlgorithmCfg,
  RslRlReppoCriticCfg,
  RslRlReppoRunnerCfg,
)


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def unitree_go1_reppo_runner_cfg() -> RslRlReppoRunnerCfg:
  """Create a REPPO runner configuration for Unitree Go1 velocity tracking."""
  return RslRlReppoRunnerCfg(
    actor=RslRlModelCfg(
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
    ),
    critic=RslRlReppoCriticCfg(
      hidden_dims=(512, 512, 512),
      activation="swish",
      obs_normalization=True,
      num_bins=151,
      vmin=-10.0,
      vmax=50.0,
    ),
    algorithm=RslRlReppoAlgorithmCfg(
      num_learning_epochs=8,
      num_mini_batches=64,
      learning_rate=3e-4,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      target_entropy=-0.5,
      init_alpha_temp=0.1,
      init_alpha_kl=0.1,
      num_kl_samples=4,
      max_grad_norm=1.0,
    ),
    experiment_name="go1_velocity_reppo",
    save_interval=50,
    num_steps_per_env=128,
    max_iterations=10_000,
  )
