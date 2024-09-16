#!/bin/env python3

import rospy
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

bridge = CvBridge()

def image_callback(msg):
    # 将 ROS 图像消息转换为 OpenCV 格式
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")

    # 使用 YOLOv5 进行目标检测
    results = model(frame)

    # 处理检测结果并发布目标位置
    for *box, conf, cls in results.xyxy[0]:  # xyxy格式的检测结果
        if int(cls) == 39:  # 仅检测指定类别（如 'bottle' 的类别索引为 39）
            x1, y1, x2, y2 = map(int, box)  # 获取目标的边界框
            rospy.loginfo("Detected bottle at {}, {}".format(x1, y1))

            # 发布目标的位置信息
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "base_link"  # 使用合适的参考坐标系
            goal_msg.pose.position.x = 1.5  # 设置目标点前方1.5米处（假设）
            goal_msg.pose.position.y = 0.0  # 可根据检测框调整目标点y坐标
            goal_msg.pose.orientation.w = 1.0

            # 发布到 /move_base_simple/goal 话题
            goal_pub.publish(goal_msg)
            rospy.loginfo("Published goal position to /move_base_simple/goal")

if __name__ == "__main__":
    rospy.init_node("yolo_object_detection")
    
    # 发布目标点
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

    # 订阅摄像头图像话题
    rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)

    rospy.spin()
