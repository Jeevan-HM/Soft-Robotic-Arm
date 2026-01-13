cmake_minimum_required(VERSION 3.5)
project(arm_vrpn_pubsub)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
include_directories(include /usr/local/include)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cppzmq REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
add_executable(arm_mocap_pubsub src/arm_mocap_pubsub.cpp)
target_link_libraries(arm_mocap_pubsub cppzmq)
ament_target_dependencies(arm_mocap_pubsub rclcpp std_msgs geometry_msgs tf2 tf2_ros cppzmq)


install(TARGETS
  arm_mocap_pubsub
  DESTINATION lib/${PROJECT_NAME})

ament_package()
