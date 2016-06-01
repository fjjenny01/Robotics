#!/usr/bin/env python

import roslib; roslib.load_manifest('objrec_ros_integration')
from rospy_message_converter import message_converter

import rospy

import objrec_ros_integration.srv

import yaml

rospy.init_node('objrec_node')

rospy.loginfo('loading stored msg')

with open('stored_msg.yaml') as f:
    doc = yaml.load(f)

def find_objects(req):
    rospy.loginfo('receiving request')
    res = objrec_ros_integration.srv.FindObjectsResponse()

    res.object_name = doc['object_name']
    res.object_pose = [message_converter.convert_dictionary_to_ros_message('geometry_msgs/PoseStamped', d) for d in doc['object_pose']]
    res.pointcloud = [message_converter.convert_dictionary_to_ros_message('sensor_msgs/PointCloud2', d) for d in doc['pointcloud']]

    return res

def main():
    # Initialization

    rospy.loginfo('initializing spoofer')

    s = rospy.Service('/objrec_node/find_objects', objrec_ros_integration.srv.FindObjects, find_objects)
    rospy.spin()


if __name__ == '__main__':
    main()
