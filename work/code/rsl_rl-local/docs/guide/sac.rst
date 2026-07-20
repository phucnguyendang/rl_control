Soft Actor-Critic
=================

The SAC implementation uses a tanh-Gaussian actor, twin online Q networks,
Polyak target networks, automatic entropy-temperature tuning, and episode-aware
n-step replay. ``OnPolicyRunner`` automatically detects replay-based algorithms,
so existing integrations that always instantiate that runner (including Mjlab)
do not need a simulator-side runner change. ``OffPolicyRunner`` is also available
for integrations that select the runner directly.

Minimal configuration
---------------------

.. code-block:: python

   train_cfg = {
       "num_steps_per_env": 16,
       "start_training": 10,
       "save_interval": 100,
       "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
       "actor": {
           "class_name": "SACActorModel",
           "hidden_dims": [256, 256, 256],
           "activation": "elu",
           "action_low": -1.0,
           "action_high": 1.0,
       },
       "critic": {
           "class_name": "SACCriticModel",
           "hidden_dims": [256, 256, 256],
           "activation": "elu",
       },
       "algorithm": {
           "class_name": "SAC",
           "replay_buffer_size": 1_000_000,
           "num_learning_epochs": 1,
           "num_mini_batches": 1,
           "mini_batch_size": 256,
           "actor_learning_rate": 1e-3,
           "critic_learning_rate": 1e-3,
           "alpha_learning_rate": 1e-3,
           "auto_alpha": True,
           "alpha": 0.05,
           "tau": 0.005,
           "gamma": 0.998,
           "policy_frequency": 2,
           "n_steps": 3,
           "rnd_cfg": None,
           "symmetry_cfg": None,
       },
   }

``replay_buffer_size`` is the total number of vectorized transitions and is
divided across environments. Actor bounds may be scalars or per-action lists.

Auto-reset timeout semantics
----------------------------

The environment may optionally put ``time_outs`` and ``time_outs_obs`` in the
extras dictionary. A timeout with ``time_outs_obs`` is retained and bootstraps
from that final observation. If the final observation is missing, the
timeout-causing transition is marked invalid. The immediately preceding valid
transition remains trainable and bootstraps from its already-known next
observation. Invalid replay slots are hard n-step boundaries, so sampling never
crosses into the reset episode.

True terminal transitions are retained with bootstrapping disabled.
