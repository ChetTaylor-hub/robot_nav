#!/bin/env python3

import rospy
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2

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
        # 绘制边界框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 在边界框上方显示类别和置信度
        cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 显示
        cv2.imshow("YOLOv5", frame)
        if int(cls) == 39:  # 仅检测指定类别（如 'bottle' 的类别索引为 39）
            x1, y1, x2, y2 = map(int, box)  # 获取目标的边界框
            rospy.loginfo("Detected bottle at {}, {}".format(x1, y1))

            # 计算目标的中心位置
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # 发布目标的位置信息
            target_msg = Float32MultiArray()
            target_msg.data = [x_center, y_center]
            target_pub.publish(target_msg)
            rospy.loginfo("Published target position to /target_position")

if __name__ == "__main__":
    rospy.init_node("yolo_object_detection")

    # 发布目标点
    target_pub = rospy.Publisher("/target_position", Float32MultiArray, queue_size=1)

    # 订阅摄像头图像话题
    rospy.Subscriber("/camera/image_raw", Image, image_callback, queue_size=1)

    rospy.spin()