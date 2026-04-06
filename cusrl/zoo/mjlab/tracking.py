from cusrl.environment import make_mjlab_env
from cusrl.environment.mjlab import MjlabPlayer, make_mjlab_env_config
from cusrl.preset import ppo
from cusrl.zoo.registry import register_experiment

register_experiment(
    environment_name=[
        "Mjlab-Tracking-Flat-Unitree-G1",
        "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    ],
    algorithm_name="ppo",
    agent_meta_factory=ppo.AgentFactory,
    agent_meta_factory_kwargs=dict(
        num_steps_per_update=24,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        normalize_observation=True,
        value_loss_weight=1.0,
        value_loss_clip=0.2,
        entropy_loss_weight=0.005,
        grad_clip_groups={"actor": 1.0, "critic": 1.0},
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_mjlab_env,
    training_env_config_factory=make_mjlab_env_config,
    player_factory=MjlabPlayer,
    playing_env_config_factory_kwargs={"play": True},
    num_iterations=30000,
    checkpoint_interval=500,
)
