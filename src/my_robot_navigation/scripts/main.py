#!/bin/env python3
import rospy
import random
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import Float32MultiArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from my_robot_navigation.msg import TargetWorldCoordinates
import tf

mapExtent = {"width": 5, "height": 5}

class RobotNavigation:
    def __init__(self):
        rospy.init_node('robot_navigation', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.target_sub = rospy.Subscriber('/yolov5/world_coordinates', BoundingBoxes, self.target_callback)
        self.bridge = CvBridge()
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.tf_listener = tf.TransformListener()
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_image = None
        self.camera_info = None
        self.target_detected = False
        self.target_position = []

    def laser_callback(self, data):
        # Check if there are obstacles within 0.5 meters in front of the robot
        min_distance = min(data.ranges)
        if min_distance < 0.3:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    
    def camera_info_callback(self, msg):
        self.camera_info = msg

    def target_callback(self, msg):
        # 先将目标坐标存上
        for box in msg.bounding_boxes:
            x_center = (box.xmin + box.xmax) / 2
            y_center = (box.ymin + box.ymax) / 2
            cls = box.Class
            self.target_position.append((x_center, y_center))
            self.target_detected = True

    def move_to_target(self, position):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = position[0]
        goal.target_pose.pose.position.y = position[1]
        goal.target_pose.pose.orientation.w = 1.0
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()

    def random_walk(self):
        # Generate random target position within a certain range
        random_position = (random.uniform(-mapExtent["width"]/2, mapExtent["width"]/2),
                           random.uniform(-mapExtent["height"]/2, mapExtent["height"]/2))
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = random_position[0]
        goal.target_pose.pose.position.y = random_position[1]
        goal.target_pose.pose.orientation.w = 1.0
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.target_detected:
                self.move_to_target(self.target_position)
                self.target_detected = False
            else:
                self.random_walk()
            rate.sleep()

if __name__ == '__main__':
    try:
        robot_nav = RobotNavigation()
        robot_nav.run()
    except rospy.ROSInterruptException:
        pass