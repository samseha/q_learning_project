#!/usr/bin/env python3

from email.mime import image
import math
import cv2

# import the moveit_commander, which allows us to control the arms
import moveit_commander
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Image
import cv_bridge

from utils import process_range_data


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
        self.pub_img = rospy.Publisher("/ar_tags", Image, queue_size=1)
        self.twist = Twist(linear=Vector3(), angular=Vector3())
        self.ang_th = 30

        self.tgt_dist = 0.235
        self.tgt_deg = -4
        self.k_dist = -0.5
        self.k_deg = -0.02
        self.k = 7
        # 0: moving to object
        # 1: grabbing object
        # 2: going to AR tag
        self.state = 2
        self.deg = None
        self.dist = None
        self.bridge = cv_bridge.CvBridge()
        self.initialized = True

    def scan_callback(self, data):
        if not self.initialized:
            return
        if self.state != 0:
            return
        retval = process_range_data(data, self.k)
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
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_points = cv2.aruco.detectMarkers(gray, aruco_dict)
        for corner_set in corners:
            corner_set = corner_set.astype(int)
            cv2.line(img, tuple(corner_set[0, 0]), tuple(corner_set[0, 1]), (255, 0, 0), 3)
            cv2.line(img, tuple(corner_set[0, 1]), tuple(corner_set[0, 2]), (255, 0, 0), 3)
            cv2.line(img, tuple(corner_set[0, 2]), tuple(corner_set[0, 3]), (255, 0, 0), 3)
            cv2.line(img, tuple(corner_set[0, 3]), tuple(corner_set[0, 0]), (255, 0, 0), 3)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

    def run(self):
        while not rospy.is_shutdown():
            if self.state == 0:
                if self.dist is None or self.deg is None:
                    continue
                dist_diff = self.tgt_dist - self.dist
                deg_diff = self.tgt_deg - self.deg
                if abs(dist_diff) < 0.003 and abs(deg_diff) <= 1:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    self.pub_twist.publish(self.twist)
                    self.state = 1
                    continue
                self.twist.linear.x = dist_diff * self.k_dist if abs(self.deg) < self.ang_th else 0
                # apply angular gain
                self.twist.angular.z = deg_diff * self.k_deg
                self.pub_twist.publish(self.twist)
            elif self.state == 1:
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
                rospy.sleep(3)

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
                rospy.sleep(3)
                self.state = 2

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
