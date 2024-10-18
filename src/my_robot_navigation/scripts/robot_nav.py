#!/bin/env python3
import numpy as np
import tf
import rospy
import random
import actionlib
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import Float32MultiArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from cv_bridge import CvBridge, CvBridgeError
from my_robot_navigation.msg import TargetWorldCoordinates
from yolov5_ros_msgs.msg import BoundingBoxes

mapExtent = {"width": 5, "height": 5}

class RobotNavigation:
    def __init__(self, cls: list):
        # 坐标转换类
        self.camera = DepthCameraCoordCovert('camera/depth/image_raw', '/camera/depth/camera_info')

        # 雷达
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        # TODO: 订阅YOLOv5检测到的目标坐标话题
        self.target_sub = rospy.Subscriber('/yolov5/BoundingBoxes', BoundingBoxes, self.BoundingBoxCallBack)
        self.bridge = CvBridge()

        # 移动底盘
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        # tf变换，用于获取相机坐标系到世界坐标系的变换
        self.tf_listener = tf.TransformListener()

        self.findObject = False     # 是否找到目标
        self.objectClass = ''       # 目标类别
        self.objectClassList = cls  # 目标类别列表
        self.target_position = []
        self.uv = []                # yolo检测到的目标中心坐标

    def laser_callback(self, data):
        # Check if there are obstacles within 0.5 meters in front of the robot
        min_distance = min(data.ranges)
        if min_distance < 0.3:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False
    
    def BoundingBoxCallBack(self, data):
        # yolo检测的回调函数

        if not self.findObject:
            # 待抓取目标为空，则请求输入抓取目标
            if self.objectClass == '':
                self.objectClass = input(f'object detected, please input the object you want to grasp:{objectClass}\n')
                if self.objectClass not in self.objectClassList:
                    rospy.loginfo('The object you want to grasp is not support!!!')
                    self.objectClass = ''
                    return
            object = []
            for dat in data.bounding_boxes:
                # 遍历所有目标，种类与待抓取目标相同则保存目标中心位置
                if self.objectClass == dat.Class:
                    object.append([dat.Class, (dat.xmin + dat.xmax)/2, (dat.ymin + dat.ymax)/2])
            if object != []:
                # 如果待抓取目标存在，则在目标列表随机选择一个返回
                rospy.loginfo('{} found, begin grasp!!!'.format(self.objectClass))
                _, u, v = random.choice(object)
                self.uv = [int(u), int(v)]
                self.findObject = True
            else:
                rospy.loginfo('The object you want to grasp is absent!!!')

    def Pexels2World(self):
        u, v = self.uv
        # 图像坐标转换为世界坐标
        worldFrame_pos = self.camera.Pixel2World(u, v)
        return worldFrame_pos

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
        while not rospy.is_shutdown():
            if self.findObject and self.camera.depth_ok:
                # 图像坐标转换为世界坐标
                worldFrame_pos = self.camera.Pixel2World(self.uv[0], self.uv[1])
                self.move_to_target(worldFrame_pos)
            else:
                self.random_walk()

class DepthCameraCoordCovert():
    def __init__(self, depth_topic, camera_info_topic):
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depthCallback)
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.depth_ok = False
        self.camera_info_ok = False

    def depthCallback(self, data):
        # 深度图像回调函数
        self.depth_ok = False
        self.depth_image = CvBridge().imgmsg_to_cv2(data, data.encoding)
        self.depth_ok = True

    def camera_info_callback(self, msg):
        # 相机内参回调函数
        self.camera_info_ok = False
        self.camera_info = msg
        self.camera_info_ok = True
        
    def get_CameraFrame_Pos(self, u, v, depthValue):
        # 图像系转相机系（u、v图像坐标，depthValue对应坐标的深度值）

        # # fx fy cx cy为相机内参
        # fx = 1043.99267578125
        # fy = 1043.99267578125
        # cx = 960
        # cy = 540

        # HACK: 相机内参应该从相机信息中获取
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        z = float(depthValue)
        x = float((u - cx) * z) / fx
        y = float((v - cy) * z) / fy

        return [x, y, z, 1]
    
    def get_RT_matrix(self, base_frame, reference_frame):
        # 获取base_frame到reference_frame旋转平移矩阵，通过tf变换获取
        listener = tf.TransformListener()
        i = 3 # 尝试3次，三次未获取到则获取失败
        while i!=0:
            try:
                listener.waitForTransform(base_frame, reference_frame,rospy.Time.now(), rospy.Duration(3.0))
                camera2World = listener.lookupTransform(base_frame, reference_frame, rospy.Time(0))
                break
            except:           
                pass
            i = i - 1

        T = camera2World[0]
        R = self.get_rotation_matrix(camera2World[1])
        R[0].append(0)
        R[1].append(0)
        R[2].append(0)
        R.append([0.0,0.0,0.0,1.0])
        R = np.mat(R)
        return [R,T]
    
    def get_rotation_matrix(q):
        # in TF, it is xyzw
        # xyzw方向转旋转矩阵
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        rot = [[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], 
                [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x],
                [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]]
        return rot
    
    def coordinate_transform(self, cameraFrame_pos, R, T):
        # 相机系转世界坐标系，先旋转再平移
        worldFrame_pos = R.I * np.mat(cameraFrame_pos).T 
        worldFrame_pos[0,0] = worldFrame_pos[0,0] + T[0]
        worldFrame_pos[1,0] = worldFrame_pos[1,0] + T[1]
        worldFrame_pos[2,0] = worldFrame_pos[2,0] + T[2]
        worldFrame_pos = [worldFrame_pos[0,0], worldFrame_pos[1,0], worldFrame_pos[2,0]]
        return worldFrame_pos
        
    def Pixel2World(self, u: int, v: int) -> list:
        [R, T] = self.get_RT_matrix('world','camera_color_optical_frame')
        # [ ]: self.depth_image[v,u]/1000.0 ? ? ?
        cameraFrame_pos = self.get_CameraFrame_Pos(u, v, self.depth_image[v,u]/1000.0)
        worldFrame_pos = self.coordinate_transform(cameraFrame_pos, R, T)
        return worldFrame_pos



if __name__ == '__main__':
    # 初始化ros节点
    rospy.init_node('robot_nav')
    # 实例化抓取导航类
    robotnavigation = RobotNavigation()
    # 实例化相机转换类
    camera = DepthCameraCoordCovert('camera/depth/image_raw', '/camera/depth/camera_info')
    # TODO: 处理YOLOv5检测到的目标坐标，结合 RobotNavigation 和 DepthCameraCoordCovert 类实现机器人导航