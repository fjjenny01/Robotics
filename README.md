Initialization
---------------
To start the PR2, run

roslaunch /etc/ros/robot.launch

launch the camera:
roslaunch /etc/ros/openni_head.launch

start ar_marker ros node:

roslaunch /home/rbtying/pr2_ar_tracker_indiv.launch


Object Recognition
------------------

On darcy navigate to /home/rbtying/graspit_bci_ws adn start the object recognition node

roslaunch objrec_ros_integration objrec_node.launch