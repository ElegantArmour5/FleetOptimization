from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import random
import matplotlib.pyplot as plt


class PackageDeliveryEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "package_delivery_v0"}

    def __init__(self):
        self.max_steps = 200
        self.num_packages = 5
        self.grid_size = (10, 10)
        self.num_trucks = 2
        self.truck_start_pos = (0, 0)

        self.agents = [f"truck_{i}" for i in range(self.num_trucks)]
        self.possible_agents = self.agents[:]

        self.reset_env()

        self.action_spaces = {f"truck_{i}": spaces.Discrete(5) for i in range(self.num_trucks)}

        obs_low = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs_high = np.array([
            self.grid_size[0] - 1,
            self.grid_size[1] - 1,
            self.num_packages,
            self.num_packages,
            self.num_packages,
            100,
            100,
            24
        ], dtype=np.float32)

        self.observation_spaces = {f"truck_{i}": spaces.Box(obs_low, obs_high, dtype=np.float32) for i in range(self.num_trucks)}

    def reset_env(self):
        self.steps = 0
        self.fuel = {f"truck_{i}": 100 for i in range(self.num_trucks)}
        self.truck_pos = {f"truck_{i}": list(self.truck_start_pos) for i in range(self.num_trucks)}
        self.time = 0
        self.package_locations = {(i + 1): (np.random.randint(10), np.random.randint(10)) for i in range(self.num_packages)}
        self.drop_locations = {(i + 1): (np.random.randint(10), np.random.randint(10)) for i in range(self.num_packages)}
        self.loads = {f"truck_{i}": [] for i in range(self.num_trucks)}
        self.packages_left = self.num_packages
        self.assign_packages()

    def assign_packages(self):
        pkg_ids = list(self.package_locations.keys())
        np.random.shuffle(pkg_ids)
        for i, pkg_id in enumerate(pkg_ids):
            truck_id = f"truck_{i % self.num_trucks}"
            self.loads[truck_id].append(pkg_id)

    def reset(self, seed=None, options=None):
        self.reset_env()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = {}
        for i in range(self.num_trucks):
            truck_id = f"truck_{i}"
            obs[truck_id] = np.array([
                self.truck_pos[truck_id][0],
                self.truck_pos[truck_id][1],
                self.packages_left,
                len(self.drop_locations),
                len(self.package_locations),
                self.fuel[truck_id],
                len(self.loads[truck_id]),
                self.time
            ], dtype=np.float32)
        return obs

    def step(self, actions):
        rewards = {}
        dones = {}
        truncations = {}
        infos = {}

        for i in range(self.num_trucks):
            truck_id = f"truck_{i}"
            action = actions[truck_id]
            reward = -1

            if action == 0:
                self.truck_pos[truck_id][1] = max(0, self.truck_pos[truck_id][1] - 1)
            elif action == 1:
                self.truck_pos[truck_id][1] = min(self.grid_size[1] - 1, self.truck_pos[truck_id][1] + 1)
            elif action == 2:
                self.truck_pos[truck_id][0] = min(self.grid_size[0] - 1, self.truck_pos[truck_id][0] + 1)
            elif action == 3:
                self.truck_pos[truck_id][0] = max(0, self.truck_pos[truck_id][0] - 1)
            elif action == 4:
                to_remove = None
                for pkg in self.loads[truck_id]:
                    if tuple(self.truck_pos[truck_id]) == self.drop_locations[pkg]:
                        reward += 50
                        to_remove = pkg
                        break
                if to_remove:
                    self.loads[truck_id].remove(to_remove)
                    self.packages_left -= 1
                    if self.steps < 150:
                        reward += 10
                else:
                    reward -= 10

            self.fuel[truck_id] -= 1
            done = self.fuel[truck_id] <= 0 or self.steps >= self.max_steps or self.packages_left == 0
            rewards[truck_id] = reward
            dones[truck_id] = done
            truncations[truck_id] = self.steps >= self.max_steps
            infos[truck_id] = {}

        self.steps += 1
        self.time = (self.time + 1) % 24

        all_done = all(dones.values())
        for truck_id in dones:
            dones[truck_id] = all_done
            truncations[truck_id] = all_done

        obs = self._get_obs()
        return obs, rewards, dones, truncations, infos

    def render(self):
        grid = np.zeros(self.grid_size)
        for truck_id, pos in self.truck_pos.items():
            grid[pos[1], pos[0]] = 1
        for pkg, pos in self.package_locations.items():
            grid[pos[1], pos[0]] = 0.5
        plt.imshow(grid, cmap='Blues')
        plt.title(f"Step: {self.steps} | Time: {self.time}h")
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()

    def close(self):
        pass


def env():
    return PackageDeliveryEnv()

register(
    id="package_delivery-v0",
    entry_point="__main__:env"
)

if __name__ == "__main__":
    from collections import defaultdict

    env_instance = env()
    env_instance = parallel_to_aec(env_instance)
    q_table = defaultdict(lambda: np.zeros(5))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2

    for episode in range(10):
        env_instance.reset()
        rewards = {agent: 0 for agent in env_instance.agents}
        for agent in env_instance.agent_iter():
            if agent not in env_instance.agents:
                continue

            obs, reward, termination, truncation, info = env_instance.last()
            if obs is None:
                continue
            state = tuple(obs.astype(int))

            if termination or truncation:
                action = None
            else:
                if random.random() < epsilon:
                    action = env_instance.action_space(agent).sample()
                else:
                    action = np.argmax(q_table[(agent, state)])

            env_instance.step(action)
            env_instance.env.render()

            if agent not in env_instance.agents:
                continue

            next_obs, _, _, _, _ = env_instance.last()
            if next_obs is not None:
                next_state = tuple(next_obs.astype(int))
                if not termination and not truncation:
                    q_table[(agent, state)][action] += alpha * (reward + gamma * np.max(q_table[(agent, next_state)]) - q_table[(agent, state)][action])

            rewards[agent] += reward

        print(f"Episode {episode + 1}, Total Rewards: {rewards}")
