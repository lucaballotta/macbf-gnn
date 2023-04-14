import os
import numpy as np
import wandb

from typing import Tuple
from time import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from macbf_gnn.env import MultiAgentEnv
from macbf_gnn.algo import Algorithm


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            log_dir: str
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.log_dir = log_dir

        # make dir for the models
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.model_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # set up log writer
        wandb.init(project='macbf-gnn')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def train(self, steps: int, eval_interval: int, eval_epi: int):
        """
        Start training

        Parameters
        ----------
        steps: int,
            number of training steps
        eval_interval: int,
            interval of steps between evaluations
        eval_epi: int,
            number of episodes for evaluation
        """
        # record start time
        start_time = time()

        # reset the environment
        data = self.env.reset()

        verbose = None
        for step in tqdm(range(1, steps + 1), ncols=80):
            action = self.algo.step(data, prob=1 - (step - 1) / steps)
            next_data, reward, done, info = self.env.step(action)
            if done:
                data = self.env.reset()
            else:
                data = next_data

            # update the algorithm
            if self.algo.is_update(step):
                verbose = self.algo.update(step, self.writer)

            # evaluate the algorithm
            if step % eval_interval == 0:
                if eval_epi > 0:
                    reward, eval_info = self.eval(step, eval_epi)
                    tqdm.write(f'step: {step}, reward: {reward:.2f}, time: {time() - start_time:.0f}s')
                if verbose is not None:
                    verbose_update = f'step: {step}'
                    for key in verbose.keys():
                        verbose_update += f', {key}: {verbose[key]:.3f}'
                    tqdm.write(verbose_update)
                self.algo.save(os.path.join(self.model_dir, f'step_{step}'))

        print(f'> Done in {time() - start_time:.0f} seconds')

    def eval(self, step: int, eval_epi: int) -> Tuple[float, dict]:
        """
        Evaluate the current model

        Parameters
        ----------
        step: int,
            current training step
        eval_epi: int,
            number of episodes for evaluation

        Returns
        -------
        reward: float
            average episode reward
        info: dict
            other information
        """
        rewards = []
        for i_epi in range(eval_epi):
            data = self.env_test.reset()
            epi_reward = 0.

            while True:
                action = self.algo.act(data)
                data, reward, done, _ = self.env_test.step(action)
                epi_reward += reward
                if done:
                    break

            rewards.append(epi_reward)

        self.writer.add_scalar('reward/test', np.mean(rewards), step)

        return np.mean(rewards).item(), {}
