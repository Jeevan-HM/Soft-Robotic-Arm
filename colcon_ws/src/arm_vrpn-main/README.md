# px4_vrpn
This is a ROS2 package that streams the VRPN data from Optitrack to PX4 using the appropriate transformations.  

Package dependencies are _vrpn_client_ros_ and _px4_ros_com._ 

Clone the repo under the `src` folder of your _colcon_ws_

To build run:
`cd <path/to/colcon_ws>`
and 
`colcon build --pacakegs-select px4_vrpn_pubsub`

To run the node:
`cd <path/to/colcon_ws>`
and then
`ros2 run px4_vrpn_pubsub mocap_pubsub`
