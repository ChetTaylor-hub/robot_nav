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
from my_robot_navigation.msg import BoundingBoxes, BoundingBox
import tf

mapExtent = {"width": 5, "height": 5}

class RobotNavigation:
    def __init__(self):
        rospy.init_node('robot_navigation', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.target_sub = rospy.Subscriber('/yolov5/targets', BoundingBoxes, self.target_callback)
        self.bridge = CvBridge()
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.tf_listener = tf.TransformListener()
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_image = None
        self.camera_info = None
        self.target_detected = False
        self.target_position = None

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
        if self.depth_image is None or self.camera_info is None:
            return

        try:
            target = msg.bounding_boxes[0]
            # 获取目标的二维坐标
            x = (target.xmin + target.xmax) / 2
            y = (target.ymin + target.ymax) / 2

            # 获取深度值
            depth = self.depth_image[int(y), int(x)]

            # 将二维坐标和深度值转换为三维坐标
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]

            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth

            # 创建目标的三维坐标
            target_camera_frame = PoseStamped()
            target_camera_frame.header.frame_id = "camera"
            target_camera_frame.header.stamp = rospy.Time.now()
            target_camera_frame.pose.position.x = X
            target_camera_frame.pose.position.y = Y
            target_camera_frame.pose.position.z = Z
            target_camera_frame.pose.orientation.w = 1.0

            # 等待转换可用
            self.tf_listener.waitForTransform("camera", "map", rospy.Time(), rospy.Duration(4.0))

            # 将目标坐标转换为地图坐标系
            target_map_frame = self.tf_listener.transformPose("map", target_camera_frame)
            self.target_position = (target_map_frame.pose.position.x, target_map_frame.pose.position.y)
            self.target_detected = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF Exception: %s", e)

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
        self.move_base_client.wait_for_result(rospy.Duration(5))

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