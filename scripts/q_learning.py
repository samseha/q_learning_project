#!/usr/bin/env python3

import csv
import os

import numpy as np
import rospy
from q_learning_project.msg import (QLearningReward, QMatrix, QMatrixRow,
                                    RobotMoveObjectToTag)
from std_msgs.msg import Header

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"


class QLearning(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("q_learning")

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-9 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][12] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt").astype(int)

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "pink", tag: 1}
        colors = ["pink", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(lambda x: {"object": colors[int(x[0])], "tag": int(x[1])}, self.actions))
        self.num_actions = len(self.actions)

        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the pink, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0, 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))
        self.num_states = len(self.states)

        self.alpha = 1
        self.gamma = 0.8
        self.current_state = 0
        self.transition_matrix = np.full((self.num_states, self.num_actions), -1)
        self.q_matrix = np.zeros_like(self.transition_matrix, dtype=float)
        self.reward = None
        self.num_steps = 300
        self.tol = 0.5

        self.action_pub = rospy.Publisher('/q_learning/robot_action', RobotMoveObjectToTag, queue_size=10)
        self.q_matrix_pub = rospy.Publisher('/q_learning/q_matrix', QMatrix, queue_size=10)
        rospy.Subscriber('/q_learning/reward', QLearningReward, self.receive_reward)

        rospy.sleep(1)
        self.initialize_q_matrix()
        self.compute_transition_matrix()

    def initialize_q_matrix(self):
        for i in range(self.num_states):
            valid_actions = np.unique(self.action_matrix[i])
            for j in range(self.num_actions):
                if j not in valid_actions:
                    self.q_matrix[i, j] = -1
        self.publish_q_matrix()

    def compute_transition_matrix(self):
        for i in range(self.num_states):
            for j in range(self.num_states):
                a = self.action_matrix[i, j]
                if a != -1:
                    self.transition_matrix[i, a] = j

    def publish_q_matrix(self):
        q_matrix = []
        for row in self.q_matrix:
            q_matrix.append(QMatrixRow(q_matrix_row=list(row.astype(int))))
        self.q_matrix_pub.publish(QMatrix(Header(stamp=rospy.Time.now()), q_matrix))

    def save_q_matrix(self):
        # save q_matrix to a file once it is done to avoid retraining
        path = os.path.dirname(__file__) + '/q_matrix.csv'
        with open(path, 'w') as f:
            csv.writer(f).writerows(self.q_matrix)

    def receive_reward(self, data):
        self.reward = data.reward

    def train_q_matrix(self):
        counter = 0
        while counter < self.num_steps:
            valid_actions = np.unique(self.action_matrix[self.current_state])[1:]
            if not len(valid_actions):
                self.current_state = 0
                continue
            a = np.random.choice(valid_actions)
            a_dict = self.actions[a]
            action_msg = RobotMoveObjectToTag(a_dict['object'], a_dict['tag'])
            self.action_pub.publish(action_msg)
            while self.reward is None:
                continue
            next_state = self.transition_matrix[self.current_state, a]
            max_q = np.max(self.q_matrix[next_state])
            diff = self.alpha * (self.reward + self.gamma * max_q - self.q_matrix[self.current_state, a])
            if diff < self.tol:
                counter += 1
            else:
                counter = 0
            self.q_matrix[self.current_state, a] += diff
            self.publish_q_matrix()
            self.current_state = next_state
            self.reward = None
        self.save_q_matrix()


if __name__ == "__main__":
    QLearning().train_q_matrix()
