#include <ros/ros.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <gazebo_msgs/LinkStates.h>
#include <gazebo_msgs/LinkState.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/ModelState.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <sensor_msgs/LaserScan.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int64.h>
#include "torch/torch.h"
#include "torch/script.h"
#include <limits>

using namespace std;
torch::Device device(torch::kCUDA);
torch::Device cpuDevice(torch::kCPU);

bool turningLeft = false;
class OverTakingWithNN{
  private:
    ros::NodeHandle n;
    ros::Publisher drive_pub;
    ros::Subscriber scan_sub;
    image_transport::ImageTransport image_transport;
    image_transport::Subscriber image_sub_;
    torch::jit::script::Module laneFollowModule;
    torch::jit::script::Module rightTurnModule;
    torch::jit::script::Module leftTurnModule;
    torch::jit::script::Module laneStateModule;
    torch::jit::script::Module turningDirectionModule;
    std::string carName = "";
    double speed = 0;
    std::vector<int> overtakingLane;
    std::vector<int> previousLane;
    std::vector<float> filteredLidarReadings;
    sensor_msgs::LaserScanConstPtr lidar;
    cv::Mat currentSemanticSegmentationImage;
    long timeOfLastCall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int drivingState = 0; 
    // States: 
    // 0 = normal driving
    // 1 = merging into side lane 
    // 2 = in side lane
    // 3 = merging back into normal lane
    long actionTimer = 0;
    bool canTurnBackIntoLane = false;
    bool worldRunning = false;
    bool hasCrossedLine = false;
    double maxSpeed = 2;
  public:
    OverTakingWithNN():image_transport(n){
        n = ros::NodeHandle("~");
        std::string drive_topic, scan_topic,laneFollowPath,rightTurnPath,leftTurnPath,laneStatePath,turningDirectionPath;
        n.getParam("nav_drive_topic", drive_topic);
        n.getParam("scan_topic", scan_topic);
        n.getParam("car_name",carName);
        n.getParam("speed",speed);
        n.getParam("/lane_follow_path",laneFollowPath);
        n.getParam("/right_turn_path",rightTurnPath);
        n.getParam("/left_turn_path",leftTurnPath);
        n.getParam("/lane_state_path",laneStatePath);
        n.getParam("/turning_direction_path",turningDirectionPath);
        drive_pub = n.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);
        scan_sub = n.subscribe(scan_topic, 1, &OverTakingWithNN::scan_callback,this);
        image_sub_ = image_transport.subscribe("/semantic_segmentation/car_output", 1, &OverTakingWithNN::image_callback, this);
        //load pytorch models
        try {
          laneFollowModule = torch::jit::load(laneFollowPath);
          laneFollowModule.to(device);
          laneFollowModule.eval();
          leftTurnModule = torch::jit::load(leftTurnPath);
          leftTurnModule.to(device);
          leftTurnModule.eval();
          rightTurnModule = torch::jit::load(rightTurnPath);
          rightTurnModule.to(device);
          rightTurnModule.eval();
          laneStateModule = torch::jit::load(laneStatePath);
          laneStateModule.to(device);
          laneStateModule.eval();
          turningDirectionModule = torch::jit::load(turningDirectionPath);
          turningDirectionModule.to(device);
          turningDirectionModule.eval();
          }
          catch (const c10::Error& e) {
            ROS_ERROR("Error Loading Model: %s", e.what());
          }
    }
    ~OverTakingWithNN()
    {
    }
    int getTurningDireciton(){
      if(currentSemanticSegmentationImage.size[0]>0){
        torch::Tensor semantic_segmentation_tensor = torch::from_blob(currentSemanticSegmentationImage.data, {1,currentSemanticSegmentationImage.rows, currentSemanticSegmentationImage.cols}, at::kByte).to(device).toType(c10::kFloat).mul(1.0/256);
          //Steering Angle
          std::vector<torch::jit::IValue> inputs;
          inputs.push_back(semantic_segmentation_tensor.toType(c10::kFloat).unsqueeze(0));
          at::Tensor outputPrediction = turningDirectionModule.forward(inputs).toTensor().to(cpuDevice);
          
          auto tensorValues = outputPrediction.accessor<float,2>();
          double maxValue = std::numeric_limits<float>::lowest();
          int maxIndex = 0;
          for(int i = 0; i < tensorValues.size(1); i++) {
            double valueAtIndex = tensorValues[0][i];
            bool test = valueAtIndex >= maxValue;
            if((valueAtIndex >= maxValue)){
              maxValue= valueAtIndex;
              maxIndex = i;
            }
          }
          return maxIndex;
      }
      return 0;
    }
    int getLaneState(){
      if(currentSemanticSegmentationImage.size[0]>0){
      torch::Tensor semantic_segmentation_tensor = torch::from_blob(currentSemanticSegmentationImage.data, {1,currentSemanticSegmentationImage.rows, currentSemanticSegmentationImage.cols}, at::kByte).to(device).toType(c10::kFloat).mul(1.0/256);
        //Steering Angle
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(semantic_segmentation_tensor.toType(c10::kFloat).unsqueeze(0));
        at::Tensor outputPrediction = laneStateModule.forward(inputs).toTensor().to(cpuDevice);
        
        auto tensorValues = outputPrediction.accessor<float,2>();
        double maxValue = std::numeric_limits<float>::lowest();;
        int maxIndex = 0;
        for(int i = 0; i < tensorValues.size(1); i++) {
          if(tensorValues[0][i]>maxValue){
            maxValue=tensorValues[0][i];
            maxIndex = i;
          }
        }
        return maxIndex;
      }
      return 0;
    }
    double getNNSteeringAngle(){ 
      if(currentSemanticSegmentationImage.size[0]>0){
        torch::Tensor semantic_segmentation_tensor = torch::from_blob(currentSemanticSegmentationImage.data, {1,currentSemanticSegmentationImage.rows, currentSemanticSegmentationImage.cols}, at::kByte).to(device).toType(c10::kFloat).mul(1.0/256);
        //Steering Angle
        std::vector<torch::jit::IValue> inputsToLaneFollow;
        inputsToLaneFollow.push_back(semantic_segmentation_tensor.toType(c10::kFloat).unsqueeze(0));
        at::Tensor outputSteeringAngle = laneFollowModule.forward(inputsToLaneFollow).toTensor();
        return outputSteeringAngle[0].item().toDouble();
      }
      return 0;
    }
    double getLeftTurningNNSteeringAngle(){ 
      if(currentSemanticSegmentationImage.size[0]>0){
        torch::Tensor semantic_segmentation_tensor = torch::from_blob(currentSemanticSegmentationImage.data, {1,currentSemanticSegmentationImage.rows, currentSemanticSegmentationImage.cols}, at::kByte).to(device).toType(c10::kFloat).mul(1.0/256);
        //Steering Angle
        std::vector<torch::jit::IValue> inputsToLaneFollow;
        inputsToLaneFollow.push_back(semantic_segmentation_tensor.toType(c10::kFloat).unsqueeze(0));
        at::Tensor outputSteeringAngle = leftTurnModule.forward(inputsToLaneFollow).toTensor();
        return outputSteeringAngle[0].item().toDouble();
      }
      return 0;
    }
    double getRightTurningNNSteeringAngle(){ 
      if(currentSemanticSegmentationImage.size[0]>0){
        torch::Tensor semantic_segmentation_tensor = torch::from_blob(currentSemanticSegmentationImage.data, {1,currentSemanticSegmentationImage.rows, currentSemanticSegmentationImage.cols}, at::kByte).to(device).toType(c10::kFloat).mul(1.0/256);
        //Steering Angle
        std::vector<torch::jit::IValue> inputsToLaneFollow;
        inputsToLaneFollow.push_back(semantic_segmentation_tensor.toType(c10::kFloat).unsqueeze(0));
        at::Tensor outputSteeringAngle = rightTurnModule.forward(inputsToLaneFollow).toTensor();
        return outputSteeringAngle[0].item().toDouble();
      }
      return 0;
    }
    bool isCarInLane(){
      if(getLaneState() == 0){
        return true;
      }
      if(getLaneState() == 1){
        return false;
      }
      return false;
    }
    double safeSpeedToFollowAt(){
      // naive version that needs to be improved
      if(lidar != nullptr){
        double minDistance = lidar->range_max;
        for(int i = 0; i < static_cast<int>(filteredLidarReadings.size());i++){
          if(filteredLidarReadings.at(i)<minDistance){
            minDistance = filteredLidarReadings.at(i);
          }
        }
        return min(sqrt(2*1.7*std::max(0.0,minDistance-0.5)),maxSpeed); // 1.7 is max acccleration of the car
      }
      return 0;
    }
    void followRoad(){
      double desiredSteeringAngle = 0;
      double desiredSpeed = this->speed;
      // int laneState = getLaneState();
      ROS_INFO("Driving State: %s",std::to_string(drivingState).c_str());
      ROS_INFO("Lane State: %s",std::to_string(isCarInLane()).c_str());

      if(drivingState == 0){
        desiredSteeringAngle = getNNSteeringAngle();
        desiredSpeed = safeSpeedToFollowAt();
        long currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if(desiredSpeed >=1.5){
          actionTimer = 0;
        }
        if(desiredSpeed<1.5){
          actionTimer += currentTime -timeOfLastCall;
          if(actionTimer>2000){ //5000 good for training data
            actionTimer=0;
            drivingState = 1;
            turningLeft = (getTurningDireciton() == 1);
          }
        }
      }

      if(drivingState == 1){
        if(turningLeft){
          desiredSteeringAngle = getLeftTurningNNSteeringAngle();
        }else{
          desiredSteeringAngle = getRightTurningNNSteeringAngle();
        }
        desiredSpeed = 3;
        if(isCarInLane() && hasCrossedLine){
          drivingState = 2;
          hasCrossedLine = false;
        }
        else if(!isCarInLane()){
          hasCrossedLine = true;
        }
      }

      if(drivingState == 2){
        desiredSteeringAngle = getNNSteeringAngle();
        //desiredSteeringAngle = steering_angle_from_points(currentCarLocation,closestPoint,secondClosestPoint);
        desiredSpeed = 2;
        long currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        actionTimer += currentTime -timeOfLastCall;
        //go back to add checking for cars in other lane
        if(actionTimer>2000){
          actionTimer=0;
          drivingState = 3;
        }
      }

      if(drivingState == 3){
        if(!turningLeft){
          desiredSteeringAngle = getLeftTurningNNSteeringAngle();
        }else{
          desiredSteeringAngle = getRightTurningNNSteeringAngle();
        }
        desiredSpeed = 3;
        if(isCarInLane() && hasCrossedLine){
          desiredSpeed = 2;
          drivingState = 0;
          hasCrossedLine = false;
        }
        else if(!isCarInLane()){
          hasCrossedLine = true;
        }
      }
      ackermann_msgs::AckermannDriveStamped drive_st_msg;
      ackermann_msgs::AckermannDrive drive_msg;
      drive_msg.steering_angle = desiredSteeringAngle;
      drive_msg.speed = desiredSpeed;
      drive_st_msg.drive = drive_msg;
      drive_pub.publish(drive_st_msg);
      timeOfLastCall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    void scan_callback(const sensor_msgs::LaserScan::ConstPtr& msg){
      lidar = msg;
      filteredLidarReadings.clear();
      for(int i = 0; i<static_cast<int>(lidar->ranges.size());i++){
          float filteredInput = lidar->ranges.at(i);
          if(!std::isfinite(filteredInput)){
            // may need to switch to zero for values lower than range min if this is a problem
            filteredInput = lidar->range_max;
          }
          filteredLidarReadings.push_back(filteredInput);
        }
    }
    void image_callback(const sensor_msgs::ImageConstPtr& msg){
      try
      {
        // for color camera
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
        cv::resize(cv_ptr->image,currentSemanticSegmentationImage,cv::Size(256,256));
        followRoad();
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
  }
};
int main(int argc, char ** argv) {
    ros::init(argc, argv, "overtaking_node", ros::init_options::AnonymousName);
    OverTakingWithNN rw;
    ros::spin();
    return 0;
}
