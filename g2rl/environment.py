from typing import Any
from copy import deepcopy
from collections import deque

import torch
import numpy as np

from pogema import pogema_v0, a_star, GridConfig
from pogema.animation import AnimationMonitor


class Grid:
    '''Basic grid container'''
    def __init__(self, obstacles: np.ndarray):
        assert obstacles.ndim == 2 and obstacles.shape[0] == obstacles.shape[1]
        self.obstacles = obstacles.copy().astype(bool)
        self.size = obstacles.shape[0]

    def is_obstacle(self, h: int, w: int) -> bool|None:
        if 0 <= h <= self.size and 0 <= w <= self.size:
            return self.obstacles[h, w]
        else:
            return False


class G2RLEnv:
    '''Environment for MAPF G2RL implementation'''
    def __init__(
            self,
            size: int = 50,
            num_agents: int = 3,
            density: float|None = None,
            map: str|list|None = None,
            obs_radius: int = 7,
            cache_size: int = 4,
            r1: float = -0.01,
            r2: float = -0.1,
            r3: float = 0.1,
            seed: int = 42,
            animation: bool = True,
            collission_system: str = 'soft',
            on_target: str = 'restart',
            max_episode_steps: int = 64,
        ):
        self.time_idx = 1
        self.num_agents = num_agents
        self.obs_radius = obs_radius
        self.cache_size = cache_size
        self.r1, self.r2, self.r3 = r1, r2, r3
        self.collission_system = collission_system
        self.on_target = on_target
        self.obs, self.info = None, None

        self._set_env(
            map,
            seed=seed,
            size=size,
            density=density,
            max_episode_steps=max_episode_steps,
            animation=animation)

        self.actions = [
            ('idle', 0, 0),
            ('up', -1, 0),
            ('down', 1, 0),
            ('left', 0, -1),
            ('right', 0, 1),
        ]

    def _get_reward(self, case: int, N: int = 0) -> float:
        rewards = [self.r1, self.r1 + self.r2, self.r1 + N * self.r3]
        return rewards[case]

    def _set_env(
            self,
            map: str|list|None,
            size: int = 48,
            density: float = 0.392,
            seed: int = 42,
            max_episode_steps: int = 64,
            animation: bool = True,
        ):
        if map is not None:
            self.grid_config = GridConfig(
                map=map,
                seed=seed,
                observation_type='MAPF',
                on_target=self.on_target,
                num_agents=self.num_agents,
                obs_radius=self.obs_radius,
                collission_system=self.collission_system,
                max_episode_steps=max_episode_steps,
            )
            self.size = self.grid_config.size
        else:
            self.grid_config = GridConfig(
                size=size,
                density=density,
                seed=seed,
                observation_type='MAPF',
                on_target=self.on_target,
                num_agents=self.num_agents,
                obs_radius=self.obs_radius,
                collission_system=self.collission_system,
                max_episode_steps=max_episode_steps,
            )
            self.size = size

        self.env = pogema_v0(grid_config=self.grid_config)
        if animation:
            self.env = AnimationMonitor(self.env)

    def _set_global_guidance(self, obs: list[dict]):
        grid = Grid(obs[0]['global_obstacles'])
        coords = [[ob['global_xy'], ob['global_target_xy']] for ob in obs]
        self.global_guidance = [a_star(st, tg, grid) for st, tg in coords]

    def save_animation(self, path):
        self.env.save_animation(path)

    def get_action_space(self) -> list[int]:
        return list(range(len(self.actions)))

    def reset(self) -> tuple[list, list]:
        self.time_idx = 1
        self.obs, self.info = self.env.reset()
        self._set_global_guidance(self.obs)
        self.view_cache = []
        for i, (ob, guidance) in enumerate(zip(self.obs, self.global_guidance)):
            guidance.remove(ob['global_xy'])
            view = self._get_local_view(ob, guidance)
            view_cache = [np.zeros_like(view) for _ in range(self.cache_size - 1)] + [view]
            self.view_cache.append(deque(view_cache, self.cache_size))
            self.obs[i]['view_cache'] = np.array(self.view_cache[-1])
        return self.obs, self.info

    def _reset_agent(self, i: int, ob: dict) -> dict[str, Any]:
        grid = Grid(ob['global_obstacles'])
        self.global_guidance[i] = a_star(ob['global_xy'], ob['global_target_xy'], grid)
        self.global_guidance[i].remove(ob['global_xy'])

        view = self._get_local_view(ob, self.global_guidance[i])
        view_cache = [np.zeros_like(view) for _ in range(self.cache_size - 1)] + [view]
        self.view_cache[i] = deque(view_cache, self.cache_size)
        ob['view_cache'] = np.array(self.view_cache[i])
        return ob

    def _get_local_view(
            self,
            obs: dict[str, Any],
            global_guidance: np.ndarray,
        ) -> np.ndarray:
        local_coord = self.obs_radius

        local_guidance = np.zeros_like(obs['agents'])
        local_size = local_guidance.shape[0]
        delta = [global_coord - local_coord for global_coord in obs['global_xy']]
        for global_cell in global_guidance:
            h = global_cell[0] - delta[0]
            w = global_cell[1] - delta[1]
            if 0 <= h < local_size and 0 <= w < local_size:
                local_guidance[h, w] = 1

        curr_agent = np.zeros_like(obs['agents'])
        curr_agent[local_coord, local_coord] = 1
        return np.dstack(
            (
                curr_agent,
                obs['obstacles'],
                obs['agents'],
                local_guidance,
            )
        )

    def step(self, actions: list[int]) -> tuple[list, ...]:
        conflict_points = set()
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # calculate reward
        for i, (action, ob, status) in enumerate(zip(actions, obs, info)):
            if status['is_active']:
                new_point = ob['global_xy']
                # conflict
                if self.actions[action] != 'idle' and new_point == self.obs[i]['global_xy']:
                    reward[i] = self._get_reward(1)
                    if self.collission_system != 'block_both':
                        # another agent (block strategy is considered)
                        if ob['global_obstacles'][new_point] == 0:
                            conflict_points.add(new_point)
                # global guidance cell
                elif new_point in self.global_guidance[i]:
                    new_point_idx = self.global_guidance[i].index(new_point)
                    reward[i] = self._get_reward(2, new_point_idx + 1)
                    # update global guidance
                    if self.on_target == 'nothing':
                        if new_point == self.global_guidance[i][-1]:
                            self.global_guidance[i] = self.global_guidance[i][-1:]
                        else:
                            self.global_guidance[i] = self.global_guidance[i][new_point_idx + 1:]
                    else:
                        self.global_guidance[i] = self.global_guidance[i][new_point_idx + 1:]
                        if len(self.global_guidance[i]) == 0:
                            ob = self._reset_agent(i, ob)
                # free cell
                else:
                    reward[i] = self._get_reward(0)

                # update history of observations
                view = self._get_local_view(ob, self.global_guidance[i])
                self.view_cache[i].append(view)

            obs[i]['view_cache'] = np.array(self.view_cache[i])

        # recalculate reward if strategy is not blocking
        if self.collission_system != 'block_both':
            for i, (ob, status) in enumerate(zip(obs, info)):
                if status['is_active'] and ob['global_xy'] in conflict_points:
                    reward[i] = self._get_reward(1)

        self.obs, self.info = obs, info
        self.time_idx += 1
        return obs, reward, terminated, truncated, info
