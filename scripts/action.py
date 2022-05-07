#!/usr/bin/env python3

import math
import cv2

# import the moveit_commander, which allows us to control the arms
import moveit_commander
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Image
import cv_bridge
import numpy as np

from utils import process_range_data
import os

path_prefix = os.path.dirname(__file__)  + "/action_states/"
q_matrix_path = os.path.dirname(__file__)  + "/q_matrix.csv"

class Robot(object):
    def __init__(self):
        self.initialized = False

        # initialize this node
        rospy.init_node('turtlebot3_action')

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_callback)
        self.pub_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub_img = rospy.Publisher("/debug", Image, queue_size=1)
        self.twist = Twist(linear=Vector3(), angular=Vector3())
        self.ang_th = 30

        self.tgt_dist = 0.235
        self.tgt_deg = -4
        self.k_dist = -0.3
        # self.k_dist = -0.1
        self.k_deg = -0.003
        # 0: identify colored object and move close to it
        # 1: move to grabbing position
        # 2: grab object
        # 3: move to AR tag
        self.state = 0
        self.deg = None
        self.dist = None
        self.bridge = cv_bridge.CvBridge()
        self.error = None
        self.initialized = True

        #import q matrix
        self.q_matrix = np.loadtxt(q_matrix_path)
        self.current_state = 0
        
        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "pink", tag: 1}
        colors = ["pink", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(lambda x: {"object": colors[int(x[0])], "tag": int(x[1])}, self.actions))
        self.num_actions = len(self.actions)

        self.transition_matrix = np.loadtxt(os.path.dirname(__file__) + "transition_matrix.txt")
        self.compute_transition_matrix()
    
    def get_action(self):
        max_score = np.max(self.q_matrix[self.current_state])
        optimal_actions = []
        for i in range(len(self.num_actions)):
            if self.q_matrix[self.current_state, i] == max_score:
                optimal_actions.append(i)
        optimal_action = np.random.choice(optimal_actions)
        dic = self.actions[optimal_action]
        self.current_state = self.transition_matrix[self.current_state, optimal_action]
        return dic["object"], dic["tag"]


    def scan_callback(self, data):
        if not self.initialized:
            return
        # if self.state not in {0, 1}:
        #     return
        retval = process_range_data(data)
        # stop if nothing in range
        # if retval is None:
        #     self.pub_twist.publish(self.stop_twist)
        #     return
        self.deg, self.dist = retval
        # print(retval)
        # apply linear gain only if angular error is small enough

    def img_callback(self, data):
        if not self.initialized:
            return
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        w = img.shape[1]
        # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # corners, ids, rejected_points = cv2.aruco.detectMarkers(gray, aruco_dict)
        # if ids is None:
        #     return
        # for id, corner_pts in zip(ids, corners):
        #     corner_pts = corner_pts[0].astype(int)
        #     print(id)
        #     cv2.line(img, tuple(corner_pts[0]), tuple(corner_pts[1]), (255, 0, 0), 3)
        #     cv2.line(img, tuple(corner_pts[1]), tuple(corner_pts[2]), (255, 0, 0), 3)
        #     cv2.line(img, tuple(corner_pts[2]), tuple(corner_pts[3]), (255, 0, 0), 3)
        #     cv2.line(img, tuple(corner_pts[3]), tuple(corner_pts[0]), (255, 0, 0), 3)
        #     self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([154, 101.4, 75.8])
        upper_pink = np.array([169, 152.6, 229.4])
        lower_cyan = np.array([91.5, 88.6, 101.4])
        upper_cyan = np.array([106.5, 139.8, 242.2])
        lower_green = np.array([31.5, 88.6, 75.8])
        upper_green = np.array([44, 152.6, 203.8])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        # mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # self.pub_img.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        M = cv2.moments(mask)
        if M['m00'] > 0:
            # center of the yellow pixels in the image
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            self.error = cx - w / 2
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
            # self.twist.angular.z = e * self.k_deg
            # self.twist_pub.publish(self.twist)
        else:
            self.error = None

    def run(self):
        while not rospy.is_shutdown():
            # print(self.state)
            if self.state == 0:
                if self.dist is None or self.deg is None:
                    continue
                if self.error is None:
                    # print('nothing detected, keep rotating')
                    self.twist.angular.z = 0.3
                    self.twist.linear.x = 0
                else:
                    self.twist.angular.z = self.error * self.k_deg
                    if abs(self.tgt_deg - self.deg) < 30:
                        if abs(self.tgt_dist - self.dist) < 0.003:
                            self.twist.linear.x = 0
                            self.twist.angular.z = 0
                            self.pub_twist.publish(self.twist)
                            self.state = 2
                            continue
                        # print('heading to object using k_dist')
                        self.twist.linear.x = (self.tgt_dist - self.dist) * self.k_dist
                        # print(self.twist.linear.x)
                    else:
                        # print('heading to object using const v')
                        self.twist.linear.x = 0.1
                        # self.twist.linear.x = 0.0
                self.pub_twist.publish(self.twist)
            # elif self.state == 1:
            #     # if self.dist is None or self.deg is None:
            #     #     continue
            #     dist_diff = self.tgt_dist - self.dist
            #     deg_diff = self.tgt_deg - self.deg
            #     if abs(dist_diff) < 0.003 and abs(deg_diff) <= 1:
            #         self.twist.linear.x = 0
            #         self.twist.angular.z = 0
            #         self.pub_twist.publish(self.twist)
            #         self.state = 2
            #         continue
            #     self.twist.linear.x = dist_diff * self.k_dist if abs(self.deg) < self.ang_th else 0
            #     # apply angular gain
            #     self.twist.angular.z = deg_diff * self.k_deg
            #     self.pub_twist.publish(self.twist)
            elif self.state == 2:
                gripper_joint_goal = [0.019, 0.019]
                self.move_group_gripper.go(gripper_joint_goal, wait=True)
                self.move_group_gripper.stop()
                rospy.sleep(1)

                arm_joint_goal = [
                    0.0,
                    math.radians(30.0),
                    math.radians(0.0),
                    math.radians(-30.0)
                ]
                self.move_group_arm.go(arm_joint_goal, wait=True)
                self.move_group_arm.stop()
                rospy.sleep(8)

                gripper_joint_goal = [0.0, 0.0]
                self.move_group_gripper.go(gripper_joint_goal, wait=True)
                self.move_group_gripper.stop()
                rospy.sleep(1)

                arm_joint_goal = [
                    0.0,
                    math.radians(-30.0),
                    math.radians(0.0),
                    math.radians(-30.0)
                ]
                self.move_group_arm.go(arm_joint_goal, wait=True)
                self.move_group_arm.stop()
                rospy.sleep(5)
                self.state = 3
            elif self.state == 3:
                arm_joint_goal = [
                    0.0,
                    math.radians(30.0),
                    math.radians(0.0),
                    math.radians(-30.0)
                ]
                self.move_group_arm.go(arm_joint_goal, wait=True)
                self.move_group_arm.stop()
                rospy.sleep(5)

                gripper_joint_goal = [0.019, 0.019]
                self.move_group_gripper.go(gripper_joint_goal, wait=True)
                self.move_group_gripper.stop()
                rospy.sleep(1)

                arm_joint_goal = [
                    0.0,
                    math.radians(-90.0),
                    math.radians(71.0),
                    math.radians(35.0)
                ]
                self.move_group_arm.go(arm_joint_goal, wait=True)
                self.move_group_arm.stop()
                # rospy.sleep(8)
                self.state = 4
        # # We can use the following function to move the arm
        # # self.move_group_arm.go(arm_joint_goal, wait=True)

        # # arm_joint_goal is a list of 4 radian values, 1 for each joint
        # # wait=True ensures that the movement is synchronous

        # # Let's move the arm based on what we have learned

        # # First determine at what angle each joint should be.
        # # You can use the GUI to find appropriate values based on your need.
        # arm_joint_goal = [
        #     0.0,
        #     math.radians(5.0),
        #     math.radians(10.0),
        #     math.radians(-20.0)
        # ]

        # # Move the arm
        # self.move_group_arm.go(arm_joint_goal, wait=True)

        # # The above should finish once the arm has fully moved.
        # # However, to prevent any residual movement,we call the following as well.
        # self.move_group_arm.stop()

        # # We can use the following function to move the gripper
        # # self.move_group_gripper.go(gripper_joint_goal, wait=True)

        # # gripper_joint_goal is a list of 2 values in meters, 1 for the left gripper and one for the right
        # # wait=True ensures that the movement is synchronous

        # # Let's move the gripper based on what we have learned

        # # First determine what how far the grippers should be from the base position.
        # # You can use the GUI to find appropriate values based on your need.
        # gripper_joint_goal = [0.00, 0.00]

        # # Move the gripper
        # self.move_group_gripper.go(gripper_joint_goal, wait=True)

        # # The above should finish once the arm has fully moved.
        # # However, to prevent any residual movement,we call the following as well.
        # self.move_group_gripper.stop()


if __name__ == '__main__':
    Robot().run()
