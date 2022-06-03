cmake_minimum_required(VERSION 3.0.2)
project(overtaking_with_neural_network)
set(Torch_DIR /home/ryan/libtorch-cpp/libtorch)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs sensor_msgs cv_bridge image_transport)

catkin_package(
	CATKIN_DEPENDS roscpp rospy
)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
 ${Torch_INCLUDE_DIRS}
)
add_executable(overtaking_with_neural_network src/overtaking_with_neural_network.cpp)
add_dependencies(overtaking_with_neural_network ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(overtaking_with_neural_network ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES}) 
