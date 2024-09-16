#!/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion

def movebase_client():
    # 初始化节点
    rospy.init_node('send_goal')

    # 创建一个 SimpleActionClient，使用 'move_base' action server
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # 等待服务器启动
    client.wait_for_server()

    # 创建目标
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # 设置目标位置和方向
    goal.target_pose.pose.position = Point(10.0, 10.0, 0.0)
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

    # 发送目标并等待结果
    client.send_goal(goal)
    wait = client.wait_for_result()

    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()

if __name__ == '__main__':
    try:
        result = movebase_client()
        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")