import numpy as np
import sys
from gym.envs.toy_text import discrete
from contextlib import closing
from io import StringIO

# define the actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Termial states are top left and
    the bottom right corner.

    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reachs a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        self.shape = (4, 4)

        self.nS = np.prod(self.shape)
        self.nA = 4

        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            P[s][UP] = self._transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._transition_prob(position, [0, 1])
            P[s][DOWN] = self._transition_prob(position, [1, 0])
            P[s][LEFT] = self._transition_prob(position, [0, -1])

        # Initial state distribution is uniform
        isd = np.ones(self.nS) / self.nS

        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(self.nS, self.nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """

        # if stuck in terminal state
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        if current_state == 0 or current_state == self.nS - 1:
            return [(1.0, current_state, 0, True)]

        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        is_done = new_state == 0 or new_state == self.nS - 1
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
