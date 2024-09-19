#! /bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import torch
import numpy as np
from functools import partial
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, RegionOfInterest
from my_robot_navigation.msg import BoundingBox, BoundingBoxes

class Yolov5Param:
    def __init__(self):
        # load local repository(YoloV5:v6.0)
        # 指定yolov5的源码路径，位于robot_vision/yolov5/
        yolov5_path = rospy.get_param('/yolov5_path', '')
        # 指定yolov5的权重文件路径，位于robot_vision/data/weights/yolov5s.pt
        weight_path = rospy.get_param('~weight_path', '')
        # yolov5的某个参数，这里不深究了
        conf = float(rospy.get_param('~conf', ''))
        # 使用pytorch加载yolov5模型，torch.hub.load会从robot_vision/yolov5/中找名为hubconf.py的文件
        # hubconf.py文件包含了模型的加载代码，负责指定加载哪个模型
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # 一个参数，用来决定使用cpu还是gpu，这里我们使用gpu
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()
        self.model.conf = 0.25

        # target publishers
        # BoundingBoxes是本样例自定义的消息类型，用来记录识别到的目标
        # 使用/yolov5/targets topic发布出去
        self.target_pub = rospy.Publisher("/yolov5/targets",  BoundingBoxes, queue_size=1)

def image_cb(msg, cv_bridge, yolov5_param, color_classes, image_pub):
    # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
    try:
        # 将Opencv图像转换numpy数组形式，数据类型是uint8（0~255）
        # numpy提供了大量的操作数组的函数，可以方便高效地进行图像处理    
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = np.array(cv_image, dtype=np.uint8)
    except CvBridgeError as e:
        print(e)
    # 实例化BoundingBoxes，存储本次识别到的所有目标信息
    bounding_boxes = BoundingBoxes()
    bounding_boxes.header = msg.header

    # 将BGR图像转换为RGB图像, 给yolov5，其返回识别到的目标信息
    # cv2.imwrite("/home/ohn/Desktop/robot_nav/src/my_robot_navigation/scripts/image/before_image.jpg", frame)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/home/ohn/Desktop/robot_nav/src/my_robot_navigation/scripts/image/after_image.jpg", rgb_image)
    results = yolov5_param.model(frame, size=640)
    boxs = results.pandas().xyxy[0].values
    for box in boxs:
        bounding_box = BoundingBox()
        # 置信度，因为是基于统计，因此每个目标都有一个置信度，标识可能性
        bounding_box.probability =np.float64(box[4])
        # （xmin, ymin）是目标的左上角，（xmax,ymax）是目标的右上角
        bounding_box.xmin = np.int64(box[0])
        bounding_box.ymin = np.int64(box[1])
        bounding_box.xmax = np.int64(box[2])
        bounding_box.ymax = np.int64(box[3])
        # 本地识别到的目标个数
        bounding_box.num = np.int16(len(boxs))
        # box[-1]是目标的类型名，比如person
        bounding_box.Class = box[-1]
        
        # 放入box队列中
        bounding_boxes.bounding_boxes.append(bounding_box)
        # 同一类目标，用同一个颜色的线条画框
        if box[-1] in color_classes.keys():
            color = color_classes[box[-1]]
        else:
            color = np.random.randint(0, 183, 3)
            color_classes[box[-1]] = color
    
        # 用框把目标圈出来
        cv2.rectangle(cv_image, (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)    
		# 在框上, 打印物体类型信息Class
        if box[1] < 20:
            text_pos_y = box[1] + 30
        else:
            text_pos_y = box[1] - 10    
        cv2.putText(cv_image, box[-1],
                    (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)    

    # 发布目标数据，topic为：/yolov5/targets
    # 可以使用命令查看：rotopic echo /yolov5/targets
    yolov5_param.target_pub.publish(bounding_boxes)
    # 将标识了识别目标的图像转换成ROS消息并发布
    image_pub.publish(cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main():
    rospy.init_node("yolov5_detector")
    rospy.loginfo("starting yolov5_detector node")

    bridge = CvBridge()
    yolov5_param = Yolov5Param()
    color_classes = {}
    image_pub = rospy.Publisher("/yolov5/detection_image", Image, queue_size=1)
    bind_image_cb = partial(image_cb, cv_bridge=bridge, yolov5_param=yolov5_param, color_classes=color_classes, image_pub=image_pub)
    rospy.Subscriber("/camera/image_raw", Image, bind_image_cb)
   
    rospy.spin()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
