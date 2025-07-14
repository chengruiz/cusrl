from cusrl.environment import make_gym_env, make_gym_vec
from cusrl.preset import ppo
from cusrl.zoo.registry import register_experiment

__all__ = []

register_experiment(
    environment_name="BipedalWalker-v3",
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=2048,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(64, 64),
        activation_fn="Tanh",
        lr=3e-4,
        sampler_epochs=4,
        sampler_mini_batches=16,
        orthogonal_init=False,
        normalize_observation=True,
        popart_alpha=0.01,
        gae_gamma=0.999,
        gae_lamda=0.95,
        entropy_loss_weight=0.0,
        max_grad_norm=0.5,
        desired_kl_divergence=0.01,
    ),
    training_env_factory=make_gym_vec,
    training_env_kwargs={"num_envs": 16},
    playing_env_factory=make_gym_env,
    playing_env_kwargs={"render_mode": "human"},
    num_iterations=400,
    save_interval=50,
)

register_experiment(
    environment_name="BipedalWalkerHardcore-v3",
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=2048,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(64, 64),
        activation_fn="Tanh",
        lr=3e-4,
        sampler_epochs=4,
        sampler_mini_batches=16,
        orthogonal_init=False,
        normalize_observation=True,
        popart_alpha=0.01,
        gae_gamma=0.999,
        gae_lamda=0.95,
        entropy_loss_weight=0.0,
        max_grad_norm=0.5,
        desired_kl_divergence=0.01,
    ),
    training_env_factory=make_gym_vec,
    training_env_kwargs={"num_envs": 16},
    playing_env_factory=make_gym_env,
    playing_env_kwargs={"render_mode": "human"},
    num_iterations=1000,
    save_interval=100,
)
