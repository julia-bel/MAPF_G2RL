from pathlib import Path
from datetime import datetime
from tqdm.notebook import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pogema import AStarAgent

from g2rl import DDQNAgent, G2RLEnv, CRNNModel
from g2rl import moving_cost, detour_percentage


def get_timestamp() -> str:
    now = datetime.now()
    timestamp = now.strftime('%H-%M-%d-%m-%Y')
    return timestamp


def get_normalized_probs(x: list[float]|None, size: int) -> np.ndarray:
    x = [1] * size if x is None else x + [0] * (size - len(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def train(
        model: torch.nn.Module,
        map_settings: dict[str, dict],
        map_probs: list[float]|None,
        num_episodes: int = 300,
        batch_size: int = 32,
        decay_range: int = 1000,
        log_dir = 'logs',
        lr: float = 0.001,
        replay_buffer_size: int = 1000,
        device: str = 'cuda'
    ) -> DDQNAgent:
    timestamp = get_timestamp()
    writer = SummaryWriter(log_dir=Path(log_dir) / timestamp)
    maps = [G2RLEnv(**args) for _, args in map_settings.items()]
    map_probs = get_normalized_probs(map_probs, len(maps))
    agent = DDQNAgent(
        model,
        maps[0].get_action_space(),
        lr=lr,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    pbar = tqdm(total=num_episodes, desc='Episodes')
    for episode in range(num_episodes):
        torch.save(model.state_dict(), f'models/{timestamp}.pt')
        env = np.random.choice(maps, p=map_probs)
        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
        obs, info = env.reset()

        state = obs[target_idx]
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]
        retrain_count = 0
        scalars = {
            'Reward': 0,
            'Moving Cost': 0,
            'Detour Percentage': 0,
            'Average Loss': 0,
            'Average Epsilon': 0,
        }

        timesteps_per_episode = 50 + 10 * episode
        for timestep in range(timesteps_per_episode):
            actions = [agent.act(ob) for agent, ob in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)
            terminated[target_idx] = obs[target_idx]['global_xy'] == opt_path[-1]

            # if the target agent has finished or FOV does not contain the global guidance
            if terminated[target_idx] or (obs[target_idx]['view_cache'][-1][:,:,-1] == 0).all():
                if terminated[target_idx]:
                    scalars['Moving Cost'] = moving_cost(timestep + 1, opt_path[0], opt_path[-1])
                    scalars['Detour Percentage'] = detour_percentage(timestep + 1, len(opt_path) - 1)
                break

            agent.store(
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            state = obs[target_idx]
            scalars['Reward'] += reward[target_idx]

            if len(agent.replay_buffer) >= batch_size:
                retrain_count += 1
                scalars['Average Loss'] += agent.retrain(batch_size)
                scalars['Average Epsilon'] += round(agent.epsilon, 4)

        for name in scalars.keys():
            if 'Average' in name and retrain_count > 0:
                scalars[name] /= retrain_count

        # logging
        for name, value in scalars.items():
            writer.add_scalar(name, value, episode)
        pbar.update(1)
        pbar.set_postfix(scalars)

    writer.close()
    return agent


if __name__ == '__main__':
    # basic elements
    map_settings = {
        'regular': {
            'size': 48,
            'density': 0.392,
            'num_agents': 4,
        },
        'random': {
            'size': 48,
            'density': 0.15,
            'num_agents': 6,
        },
        'free': {
            'size': 48,
            'density': 0,
            'num_agents': 11,
        },
    }
    map_probs = [0.3, 0.35, 0.35]
    model = CRNNModel()
    device = 'cpu'
    
    # train loop
    trained_agent = train(
        model,
        map_settings=map_settings,
        map_probs=map_probs,
        num_episodes=300,
        batch_size=32,
        replay_buffer_size=500,
        decay_range=10_000,
        log_dir='logs',
        device=device,
    )
    
    # save model
    torch.save(model.state_dict(), 'models/best_model.pt')
