import json
import random
from pprint import pprint

import gym
import numpy
# from tqdm import tqdm
import tqdm.contrib.concurrent


class QLearning:
    def __init__(self, env: gym.Env):
        self.env = env
        self.Q = numpy.zeros((env.observation_space.n, env.action_space.n))

        flat_desc = [item for sublist in env.desc for item in sublist]
        rewards = {b'G': 100, b'H': -100, b'F': -100, b'S': -100}
        self.reward = [rewards[field] for field in flat_desc]

    def train(self, iters: int, beta: float, gamma: float, epsilon: float):
        # for _ in tqdm(range(iters)):
        for _ in range(iters):
            state = self.env.reset()
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = numpy.argmax(self.Q[state])

                next_state, _, done, _ = self.env.step(action)
                self.Q[state, action] += beta * (self.reward[next_state] + gamma * numpy.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state

    def evaluate(self, iters: int) -> float:
        successes = 0
        # for _ in tqdm(range(iters)):
        for _ in range(iters):
            state = self.env.reset()
            for move_index in range(200):
                action = numpy.argmax(self.Q[state])
                state, _, done, _ = self.env.step(action)
                if done:
                    successes += 1
                    break

        return successes / iters

    def clear(self):
        self.Q = numpy.zeros((self.env.observation_space.n, self.env.action_space.n))


def test(params: (float, float, float)) -> {float, float, float, float}:
    results = []
    model = QLearning(env=gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).env)
    for _ in range(25):
        model.clear()
        model.train(iters=10_000, beta=params[0], gamma=params[1], epsilon=params[2])
        results.append(model.evaluate(iters=1000))
    return {
        'beta': params[0],
        'gamma': params[1],
        'epsilon': params[2],
        'success rate': numpy.mean(results),
        'success rate std': numpy.std(results)
    }


if __name__ == '__main__':
    model = QLearning(env=gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).env)
    model.train(iters=10_000, beta=0.9, gamma=0.9, epsilon=0.9)
    print(model.evaluate(iters=1000))
    # pprint(test((0.1, 0.1, 0.1)))
    # variants = [(i / 10, j / 10, k / 10) for k in range(5, 11) for j in range(1, 11) for i in range(1, 11)]
    variants = [(k / 10, k / 10, k / 10) for k in range(1, 11)]

    result = tqdm.contrib.concurrent.process_map(test, variants, max_workers=12)

    # pprint(list(result))

    data = {'scores': list(result)}
    json_string = json.dumps(data)

    # Using a JSON string
    with open('json_data3.json', 'w') as outfile:
        outfile.write(json_string)
