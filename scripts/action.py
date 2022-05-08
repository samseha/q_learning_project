#!/usr/bin/env python3

import math
import os

import cv2
import cv_bridge
# import the moveit_commander, which allows us to control the arms
import moveit_commander
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image, LaserScan

from utils import process_range_data

path_prefix = os.path.dirname(__file__) + "/action_states/"
q_matrix_path = os.path.dirname(__file__) + "/q_matrix.csv"
t_matrix_path = os.path.dirname(__file__) + "/transition_matrix.txt"


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

        self.tgt_dist = 0.235
        self.tgt_ang = -4
        self.k_dist = -0.3
        self.k_ang = -0.003
        # 0: identify colored object and move close to it
        # 1: grab object
        # 2: move to AR tag
        # 3: Drop Object
        # 4: Finished
        self.state = 0
        self.ang = None
        self.dist = None
        self.bridge = cv_bridge.CvBridge()
        self.error = None
        self.initialized = True

        # import q matrix
        self.q_matrix = np.loadtxt(q_matrix_path, delimiter=',')
        self.current_state = 0

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "pink", tag: 1}
        colors = ["pink", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(lambda x: {"object": colors[int(x[0])], "tag": int(x[1])}, self.actions))
        self.num_actions = len(self.actions)

        self.transition_matrix = np.loadtxt(t_matrix_path).astype(int)

        self.next_object = None
        self.next_tag = None

        self.matched_object = 0
        self.lower_pink = np.array([154, 101.4, 75.8])
        self.upper_pink = np.array([169, 152.6, 229.4])
        self.lower_blue = np.array([91.5, 88.6, 101.4])
        self.upper_blue = np.array([106.5, 139.8, 242.2])
        self.lower_green = np.array([31.5, 88.6, 75.8])
        self.upper_green = np.array([44, 152.6, 203.8])

    def get_action(self):
        max_score = np.max(self.q_matrix[self.current_state])
        optimal_actions = []
        for i in range(self.num_actions):
            if self.q_matrix[self.current_state, i] == max_score:
                optimal_actions.append(i)
        optimal_action = np.random.choice(optimal_actions)
        dic = self.actions[optimal_action]
        self.current_state = self.transition_matrix[self.current_state, optimal_action]
        self.next_object = dic["object"]
        self.next_tag = dic["tag"]

    def scan_callback(self, data):
        if not self.initialized:
            return
        if self.state not in {0, 2}:
            return
        # retval = process_range_data(data, lo=-15, hi=16)
        retval = process_range_data(data, lo=self.tgt_ang - 15, hi=self.tgt_ang + 16)
        if retval is not None:
            self.ang, self.dist = retval

    def img_callback(self, data):
        if not self.initialized:
            return
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        w = img.shape[1]
        if self.state == 0:
            # finding colored object
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if self.next_object == "pink":
                mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)
            elif self.next_object == "green":
                mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            else:
                mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            # self.pub_img.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
            M = cv2.moments(mask)
            if M['m00'] > 0:
                # center of the colored pixels in the image
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.error = (cx - w / 2, rospy.Time.now())
                cv2.circle(mask, (cx, cy), 10, 255, -1)
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))
        elif self.state == 2:
            # finding AR tag
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            if ids is None:
                return
            for id, corner_pts in zip(ids, corners):
                if id == self.next_tag:
                    corner_pts = corner_pts[0].astype(int)
                    cx = np.mean(corner_pts[:, 0])
                    self.error = (cx - w / 2, rospy.Time.now())
                    # cv2.line(img, tuple(corner_pts[0]), tuple(corner_pts[1]), (255, 0, 0), 3)
                    # cv2.line(img, tuple(corner_pts[1]), tuple(corner_pts[2]), (255, 0, 0), 3)
                    # cv2.line(img, tuple(corner_pts[2]), tuple(corner_pts[3]), (255, 0, 0), 3)
                    # cv2.line(img, tuple(corner_pts[3]), tuple(corner_pts[0]), (255, 0, 0), 3)
                    cv2.polylines(img, [corner_pts], True, (255, 0, 0), 3)
                    self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
                    break

    def reset_to_middle(self):
        self.twist.angular.z = math.radians(180 / 5)
        self.pub_twist.publish(self.twist)
        rospy.sleep(5)
        self.twist.angular.z = 0
        self.pub_twist.publish(self.twist)
        self.twist.linear.x = 0.1
        self.pub_twist.publish(self.twist)
        rospy.sleep(5)
        self.twist.linear.x = 0
        self.pub_twist.publish(self.twist)

    def run(self):
        gripper_joint_goal = [0.019, 0.019]
        self.move_group_gripper.go(gripper_joint_goal)
        self.move_group_gripper.stop()

        tgt_dist = self.tgt_dist
        self.get_action()
        while not rospy.is_shutdown():
            print(self.state)
            # print(self.state)
            # 0: spin, find colored object and move to it
            # 1: grab object
            # 2: spin, find AR tag and move to it
            # 3: Drop Object
            if self.state in {0, 2}:
                if self.error is None or (rospy.Time.now() - self.error[1]).to_sec() > 0.3:
                    # print('nothing detected, keep spinning')
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0.3
                else:
                    if abs(self.error[0]) < 30 and self.dist is not None:
                        if abs(tgt_dist - self.dist) < 0.01:
                            self.twist.linear.x = 0
                            self.twist.angular.z = 0
                            self.pub_twist.publish(self.twist)
                            self.state += 1
                            continue
                        self.twist.linear.x = (tgt_dist - self.dist) * self.k_dist
                    else:
                        self.twist.linear.x = 0
                    self.twist.angular.z = self.error[0] * self.k_ang
                self.pub_twist.publish(self.twist)
            elif self.state == 1:
                # Picking up Object
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
                # reset
                self.reset_to_middle()
                tgt_dist += 0.3
                self.state += 1
            elif self.state == 3:
                # Dropping off Object
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
                # reset
                self.reset_to_middle()
                self.matched_object += 1
                if self.matched_object == 3:
                    gripper_joint_goal = [0, 0]
                    self.move_group_gripper.go(gripper_joint_goal, wait=True)
                    self.move_group_gripper.stop()
                    return
                tgt_dist -= 0.3
                self.state = 0
                self.get_action()


if __name__ == '__main__':
    Robot().run()
