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
        # filter class
        self.cls = ["car"]

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


    def fiter_class(self, fiter: list):
        cls = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24, "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29, "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34, "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38, "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68, "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73, "clock":
               74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78, "toothbrush": 79}
        
        # 从cls中删除不需要的类别
        for key in cls.keys():
            if key not in fiter:
                del cls[key]
        return cls

def image_cb(msg, cv_bridge, yolov5_param, color_classes, image_pub):
    
    try:
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式，将Opencv图像转换numpy数组形式，数据类型是uint8（0~255）
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = np.array(cv_image, dtype=np.uint8)
    except CvBridgeError as e:
        print(e)

    # 实例化BoundingBoxes，存储本次识别到的所有目标信息
    bounding_boxes = BoundingBoxes()
    bounding_boxes.header = msg.header

    # image size
    img_size = 320

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolov5_param.model(rgb_image, size=img_size)
    boxs = results.pandas().xyxy[0].values

    for box in boxs:
        # 过滤掉不需要的类别
        if box[-1] not in yolov5_param.cls:
            continue
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
    rospy.Subscriber("/camera/rgb/image_raw", Image, bind_image_cb)
   
    rospy.spin()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
