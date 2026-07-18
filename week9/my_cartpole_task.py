from __future__ import annotations
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

_CARTPOLE_XML: Path = Path(__file__).parent / "cartpole.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


# Entity.

def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg

_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)
from mjlab.entity import EntityCfg
_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)

_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi},
  joint_vel={".*": 0.0},
)

def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_CARTPOLE_ARTICULATION,
    init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT,
  )


# Observations.

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)

# Rewards
# dm_control uses value_at_margin=0.1 by default.
_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)

def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Gaussian sigmoid tolerance: 1 at x=0, value_at_margin=0.1 at |x|=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _GAUSSIAN_SCALE
  return torch.exp(-0.5 * scaled**2)

def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Quadratic sigmoid tolerance: 1 at x=0, 0 at |x|>=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _QUADRATIC_SCALE
  return torch.clamp(1 - scaled**2, min=0.0)


def cartpole_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """dm_control smooth cartpole reward: upright * centered * small_control * small_vel.

  Args:
    env: The environment.
    cart_cfg: Entity config selecting the slider joint.
    hinge_cfg: Entity config selecting the hinge joint.
  """
  asset: Entity = env.scene[cart_cfg.name]

  # Pole angle cosine.
  hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)
  pole_cos = torch.cos(hinge_angle)
  upright = (pole_cos + 1) / 2

  # Cart position.
  cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  # Control effort (raw action from the policy).
  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  # Pole angular velocity.
  hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

  return upright * centered * small_control * small_velocity


from mjlab.envs import ManagerBasedRlEnv
# Environment config.
def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
  hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

  from mjlab.managers.observation_manager import ObservationTermCfg
  from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel
  actor_terms = {
    "cart_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": cart_cfg},
    ),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin,
      params={"asset_cfg": hinge_cfg},
    ),
    "cart_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": cart_cfg},
    ),
    "pole_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": hinge_cfg},
    ),
  }

  from mjlab.managers.observation_manager import ObservationGroupCfg
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  from mjlab.envs.mdp.actions import JointEffortActionCfg
  from mjlab.managers.action_manager import ActionTermCfg
  
  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  slider_range = (-0.1,0.1) if not swing_up else (0,0)
  
  from mjlab.envs.mdp.events import reset_joints_by_offset
  from mjlab.managers.event_manager import EventTermCfg
  # define events will be running when the environment reset
  events = {
    "reset_slider": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": slider_range,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.034, 0.034),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
      },
    ),
  }

  from mjlab.managers.reward_manager import RewardTermCfg
  rewards = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_smooth_reward,
      weight=1.0,
      params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
    ),
  }
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.envs.mdp.terminations import time_out
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }
  
  
  from mjlab.viewer import ViewerConfig
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg
  from mjlab.sim import SimulationCfg, MujocoCfg
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"cartpole": _get_cartpole_cfg(swing_up=swing_up)},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="cartpole",
      body_name="cart",
      distance=4.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    decimation=5,
    episode_length_s=50.0,
  )

def cartpole_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_swingup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg

# RL config.

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
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
    experiment_name="cartpole",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )

from mjlab.tasks.registry import register_mjlab_task


register_mjlab_task(
  task_id="Local-Cartpole-Swingup",
  env_cfg=cartpole_swingup_env_cfg(),
  play_env_cfg=cartpole_swingup_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)
