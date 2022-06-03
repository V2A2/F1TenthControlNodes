cmake_minimum_required(VERSION 3.0.2)
set (CMAKE_CXX_STANDARD 17)
project(overtaking_recorder)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs sensor_msgs cv_bridge image_transport)

catkin_package(
	CATKIN_DEPENDS roscpp rospy
)
find_package(OpenCV REQUIRED)
include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
)
add_executable(overtaking_recorder src/overtaking_recorder.cpp)
add_dependencies(overtaking_recorder ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(overtaking_recorder ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}) 
