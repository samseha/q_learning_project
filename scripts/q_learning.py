#!/usr/bin/env python3

import rospy
import numpy as np
import os
from q_learning_project.msg import RobotMoveObjectToTag, QLearningReward, QMatrix, QMatrixRow
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
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "pink", tag: 1}
        colors = ["pink", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"object": colors[int(x[0])], "tag": int(x[1])},
            self.actions
        ))


        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the pink, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0 , 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        self.alpha = 1
        self.gamma = 0.8
        self.current_state = 0
        self.transition_matrix = np.full((len(self.states), len(self.actions)), -1)

        self.q_matrix = np.zeros((len(self.states), len(self.actions)))
        #self.q_matrix = [[0] * len(self.actions) for _ in range(len(self.states))]
        #self.save_q_matrix()
        self.reward = None
        self.reward_flag = False
        self.action_pub = rospy.Publisher('/q_learning/robot_action', RobotMoveObjectToTag, queue_size=10)
        self.q_matrix_pub = rospy.Publisher('/q_learning/q_matrix', QMatrix, queue_size=10)
        rospy.Subscriber('/q_learning/reward', QLearningReward, self.get_reward)
        print("go sleep")
        rospy.sleep(1)
        print("woke up")
        self.initialize_q_matrix()
        self.compute_transition_matrix()
        
    def initialize_q_matrix(self):
        for i in range(len(self.action_matrix)):
            valid_actions = np.unique(self.action_matrix[i])
            for j in range(len(self.actions)):
                if j not in valid_actions:
                    self.q_matrix[i, j] = -1
        self.publish_q_mat()

    def compute_transition_matrix(self):
        for i in range(len(self.action_matrix)):
            for j in range(len(self.action_matrix)):
                if self.action_matrix[i,j] != -1:
                    self.transition_matrix[i, int(self.action_matrix[i,j])] = j
        print(self.transition_matrix)

    def publish_q_mat(self):    
        header = Header(stamp = rospy.Time.now())
        pub_matrix = []
        for row in self.q_matrix:
            pub_matrix.append(QMatrixRow(q_matrix_row = list(row.astype(int))))
        q_mat = QMatrix(header, pub_matrix)
        self.q_matrix_pub.publish(q_mat)

    def save_q_matrix(self):
        # TODO: You'll want to save your q_matrix to a file once it is done to
        # avoid retraining
        path = os.path.dirname(__file__) + '/q_matrix.txt'
        np.savetxt(path, self.q_matrix)
        return

    def get_reward(self, data):
        self.reward = data.reward
        self.reward_flag = True
        return

    def train_q_matrix(self):
        counter = 0
        while counter < 500:
            valid_actions = np.unique(self.action_matrix[self.current_state]).astype(int)
            valid_actions = valid_actions[1:]
            if len(valid_actions) == 0:
                self.current_state = 0
                continue
            a = np.random.choice(valid_actions)
            a_dic = self.actions[a]
            next_state = self.transition_matrix[self.current_state, a]
            pub_msg = RobotMoveObjectToTag(a_dic['object'], a_dic['tag'])
            self.action_pub.publish(pub_msg)
            while not self.reward_flag:
                continue
            self.reward_flag = False
            max_q = np.max(self.q_matrix[next_state])
            diff = self.alpha * (self.reward + self.gamma * max_q - self.q_matrix[self.current_state, a])
            if diff < 0.5:
                counter += 1
            else:
                counter = 0
            self.q_matrix[self.current_state, a] += diff
            self.publish_q_mat()
            self.current_state = next_state
            print(counter)
        self.save_q_matrix()
    

if __name__ == "__main__":
    node = QLearning()
    #print(node.q_matrix)

    node.train_q_matrix()