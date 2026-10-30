"""Small CPU PyTorch DQN, explicitly not an SB3 implementation."""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np


def _torch():
    import torch
    return torch


def network(observation_size, hidden_sizes=(64, 64)):
    torch = _torch()
    layers = []
    current = observation_size
    for size in hidden_sizes:
        layers += [torch.nn.Linear(current, size), torch.nn.ReLU()]
        current = size
    layers.append(torch.nn.Linear(current, 2))
    return torch.nn.Sequential(*layers)


def train(make_env, episode_count: int, observation_size: int, seed: int, config: dict):
    """Replay DQN with finite-horizon Bellman targets and a frozen config."""
    torch = _torch()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    q = network(observation_size, config["hidden_sizes"])
    target = network(observation_size, config["hidden_sizes"])
    target.load_state_dict(q.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(q.parameters(), lr=config["learning_rate"])
    memory = deque(maxlen=config["replay_size"])
    steps = int(config["total_steps"])
    if steps < 1 or episode_count < 1:
        raise ValueError("training requires positive steps and nonempty episodes")
    env = make_env(int(rng.integers(episode_count)))
    obs, _ = env.reset(seed=seed)
    losses = []
    start = perf_counter()
    for t in range(steps):
        fraction = min(1.0, t / max(1, steps * config["epsilon_fraction"]))
        epsilon = 1.0 + fraction * (config["epsilon_end"] - 1.0)
        if rng.random() < epsilon:
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
        nxt, reward, terminated, truncated, _ = env.step(action)
        # Both flags end this configured finite-horizon task. No terminal bootstrap.
        done = terminated or truncated
        memory.append((obs.copy(), action, reward, nxt.copy(), done))
        obs = nxt
        if len(memory) >= max(config["warmup_steps"], config["batch_size"]) and t % config["train_frequency"] == 0:
            samples = [memory[int(i)] for i in rng.choice(len(memory), config["batch_size"], replace=False)]
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.int64)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)
            values = q(states).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                expected = rewards + config["gamma"] * (1.0 - dones) * target(next_states).max(dim=1).values
            loss = torch.nn.functional.smooth_l1_loss(values, expected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), config["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.item()))
        if (t + 1) % config["target_update_steps"] == 0:
            target.load_state_dict(q.state_dict())
        if done:
            env = make_env(int(rng.integers(episode_count)))
            obs, _ = env.reset(seed=seed)
    metadata = {"implementation": "native_pytorch_dqn", "seed": seed,
                "training_steps": steps, "updates": len(losses),
                "last_100_loss_mean": float(np.mean(losses[-100:])) if losses else None,
                "training_seconds": perf_counter() - start,
                "observation_size": observation_size, "config": config}
    return q.eval(), metadata


def save(path: Path, q, metadata):
    torch = _torch()
    torch.save({"state_dict": q.state_dict(), "metadata": metadata}, path)


def load(path: Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    q = network(metadata["observation_size"], metadata["config"]["hidden_sizes"])
    q.load_state_dict(payload["state_dict"])
    return q.eval(), metadata


def predict(q, obs):
    torch = _torch()
    with torch.no_grad():
        return int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
