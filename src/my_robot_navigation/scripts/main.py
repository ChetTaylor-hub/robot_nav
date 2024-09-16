#!/bin/env python3
import rospy
import random
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import cv2
from cv_bridge import CvBridge
import torch

mapExtent = {"width": 8, "height": 8}

class RobotNavigation:
    def __init__(self):
        rospy.init_node('robot_navigation', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.target_sub = rospy.Subscriber('/yolov5/targets', Float32MultiArray, self.target_callback)
        self.bridge = CvBridge()
        # self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.target_detected = False
        self.target_position = None
        self.obstacle_detected = False

    # def image_callback(self, data):
    #     cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     results = self.model(cv_image)
    #     for *box, conf, cls in results.xyxy[0]:
    #         if int(cls) == 0:  # Assuming the target is class 0 (person)
    #             self.target_detected = True
    #             self.target_position = self.calculate_target_position(box)
    #             break

    def laser_callback(self, data):
        # Check if there are obstacles within 0.5 meters in front of the robot
        min_distance = min(data.ranges)
        if min_distance < 0.3:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def target_callback(self, msg):
        self.target_detected = True
        self.target_position = (msg.data[0], msg.data[1])

    def calculate_target_position(self, box):
        # Placeholder for actual position calculation
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return (x_center, y_center)

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