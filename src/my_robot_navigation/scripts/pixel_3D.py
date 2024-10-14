import rospy  # noqa: F401
import tf
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64, Int64, Int16, String  # noqa: F401
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2  # noqa: F401
import cv_bridge
import numpy as np  # noqa: F401
from my_robot_navigation.msg import BoundingBoxes, TargetWorldCoordinates, TargetWorldCoordinate
import actionlib  # noqa: F401
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  # noqa: F401

#!/usr/bin/env python


import rospy
import tf
import numpy as np

class Pixel3DConverter:
    def __init__(self):
        rospy.init_node('pixel_3d_converter', anonymous=True)
        
        self.bridge = cv_bridge.CvBridge()

        # world coordinates pub
        self.world_coordinates_pub = rospy.Publisher('/yolov5/world_coordinates', TargetWorldCoordinates, queue_size=1)

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        
        self.target_sub = rospy.Subscriber('/yolov5/targets', BoundingBoxes, self.target_callback)
        self.depth_sub = Subscriber('/camera/depth/image_raw', Image)
        self.camera_info_sub = Subscriber('/camera/depth/camera_info', CameraInfo)
        
        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.camera_info_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.depth_camera_callback)
        
        self.camera_info = None
        self.depth_image = None

        self.tf_listener = tf.TransformListener()

    def get_camera_extrinsics(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/base_footprint', '/camera', rospy.Time(0))
            R = tf.transformations.quaternion_matrix(rot)[:3, :3]
            t = np.array(trans)
            return R, t
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("TF transform not available")
            return None, None

    def target_callback(self, msg):
        if self.depth_image is None or self.camera_info is None:
            return

        R, t = self.get_camera_extrinsics()
        if R is None or t is None:
            return

        target_world_coordinates = TargetWorldCoordinates()
        for target in msg.bounding_boxes:
            target_world_coordinate = TargetWorldCoordinate()
            u = (target.xmin + target.xmax) / 2
            v = (target.ymin + target.ymax) / 2
            depth = self.depth_image[int(v), int(u)]
            x, y, z = self.pixel_to_camera(u, v, depth)
            x, y, z = self.camera_to_world(x, y, z, R, t)

            target_world_coordinate.x = x
            target_world_coordinate.y = y
            target_world_coordinate.z = z
            target_world_coordinates.TargetWorldCoordinates.append(target_world_coordinate)

            rospy.loginfo(f"Class: {target.Class}, World Coordinates: x={x}, y={y}, z={z}")
            
            # 发送目标位置给move_base
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "base_footprint"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = z
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.position.z = x
            goal.target_pose.pose.orientation.w = 1.0  # 假设没有旋转

            self.move_base_client.send_goal(goal)
            self.move_base_client.wait_for_result()
        self.world_coordinates_pub.publish(target_world_coordinates)

    def depth_camera_callback(self, depth_msg, camera_info_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.camera_info = camera_info_msg

    def pixel_to_camera(self, u, v, depth):
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return x, y, z

    def camera_to_world(self, x, y, z, R, t):
        (trans, rot) = self.tf_listener.lookupTransform('/base_footprint', '/camera', rospy.Time(0))
        # 将旋转和平移转为4x4的齐次变换矩阵
        T_camera_to_world = self.tf_listener.fromTranslationRotation(trans, rot)

        # 将相机坐标转为齐次坐标
        P_camera = np.array([x, y, z, 1]).reshape(4, 1)

        # 计算世界坐标
        P_world = np.dot(T_camera_to_world, P_camera)
        X_world, Y_world, Z_world = P_world[:3].flatten()

        return X_world, Y_world, Z_world

if __name__ == '__main__':
    converter = Pixel3DConverter()
    rospy.spin()