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
#include <fstream>
#include <std_msgs/Int64.h>
#include <filesystem>

using namespace std;
struct Point{
    double x;
    double y;
    double z;
    double rotation;
};
class RoadSegment{
    //TODO: Write some way to convert an xml like format to RoadSegments
    /*
    0:Line calulated by having one endpoint at x,y,z and then have the other endpoint at the end of the vector centered at x,y,z with length param1 and angle param2
    1:Circle Segment calulated by taking arc from circle centered at x,y,z with radius param1, between the angles start and end
    ALL angles will be in radians
    Designed from standard xy plane view where angle 0 is to the right
    */
    public:
        //measurements in meters
        double x;
        double y;
        double z;
        int classification; // 0 = line 1 = circle
        //Paramrs 1 for both cirlce and line, params 2-4 for circle creation
        double param1; // length for line radius for cirle
        double param2; // angle for line and start angle for circle
        double param3; // end angle for circle
        RoadSegment(double x, double y, double z, int classification, double param1, double param2, double param3){
            this->x = x;
            this->y = y;
            this->z = z;
            this->classification = classification;
            this->param1 = param1;
            this->param2 = param2;
            this->param3 = param3;
        }
        double getLengthOfSegment(){
            if(classification == 0){
                return param1;
            }
            if(classification == 1){
                return abs(param1 * (param2-param3));
            }
            return 0;
        }
        Point getPointXLengthInLine(double distanceIn){
            Point point;
            point.z = z;
            if(classification == 0){
                point.x = x + distanceIn * cos(param2);
                point.y = y + distanceIn * sin(param2);
                point.rotation = param2;
            }
            if(classification ==1){
                double angleToRevolve = distanceIn / param1;
                int direction = 1;
                if(param3<param2){
                    direction = -1;
                }
                double angleOfNewPoint = param2 + angleToRevolve * direction;
                point.x = x + param1 * cos(angleOfNewPoint);
                point.y = y + param1 * sin(angleOfNewPoint);
                
                point.rotation = angleOfNewPoint+ M_PI/2;
                if(param2>param3){
                    point.rotation = angleOfNewPoint - M_PI/2;
                }

            }
            return point;
        }
};
class RoadLine{
    public:
        std::vector<RoadSegment> roadSegments;
        bool dotted = false;
        bool yellowLine = false;
        int visualTagIndex = 0;
        std::vector<Point> points;
        double default_resolution_for_point_map = 0.01;
        RoadLine(std::vector<RoadSegment>segments,int elementTag){
            this->roadSegments = segments;
            dotted = false;
            yellowLine = false;
            visualTagIndex = elementTag;
            this->points = getRoadPoints(default_resolution_for_point_map);
        }
        RoadLine(std::vector<RoadSegment>segments,bool isDotted, bool isYellowLine, int elementTag){
            this->roadSegments = segments;
            dotted = isDotted;
            yellowLine = isYellowLine;
            visualTagIndex = elementTag;
            this->points = getRoadPoints(default_resolution_for_point_map);
        }
        std::vector<Point> getRoadPoints(double resolution){ // resolution is distance between points in meters
            std::vector<Point> listOfPoints;
            double lastDistance = 0;
            for(int i=0;i<static_cast<int>(roadSegments.size());i++){
                RoadSegment currentSegment = roadSegments.at(i);
                double currentLength = lastDistance;
                while(currentLength<=currentSegment.getLengthOfSegment()){
                    listOfPoints.push_back(currentSegment.getPointXLengthInLine(currentLength));
                    currentLength+=resolution;
                }
                lastDistance = currentLength -currentSegment.getLengthOfSegment();
            }
            return listOfPoints;
        }
        std::string getRoadSDFString(){
            if(this->dotted && this->yellowLine){
                return getRoadSDFStringDottedYellowLine();
            }
            if(this->dotted && !this->yellowLine){
                return getRoadSDFStringDottedWhiteLine();
            }
            if(!this->dotted && this->yellowLine){
                return getRoadSDFStringYellowLine();
            }
            return getRoadSDFStringWhiteLine();
        }
        Point getClosestPointOnRoad(Point referencePoint){
          Point closestPoint= this->points.at(0);
          double minDistanceToPoint = distanceBetweenPoints(closestPoint,referencePoint);
          for(int i = 0;i<static_cast<int>(this->points.size());i++){
            if(distanceBetweenPoints(referencePoint,this->points.at(i))<minDistanceToPoint){
              minDistanceToPoint = distanceBetweenPoints(referencePoint,this->points.at(i));
              closestPoint = this->points.at(i);
            }
          }
          return closestPoint;
        }

        double distanceToClosestPoint(Point reference){
          return distanceBetweenPoints(reference,getClosestPointOnRoad(reference));
        }

        std::string getRoadSDFStringYellowLine(){
            std::string roadString = "";
            std::vector<Point> pointsToDisplay = getRoadPoints(0.1);
            for (int index = 0; index<static_cast<int>(pointsToDisplay.size());index++){
                Point point = pointsToDisplay.at(index);
                roadString+="<visual name ='visual"+ std::to_string(this->visualTagIndex)+"-"+std::to_string(index)+"'><pose>";
                roadString+= "" +std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z) + " 0 0 " +std::to_string(point.rotation);
                roadString+="</pose><geometry><plane><normal>0 0 1</normal> <size>0.1 0.03</size></plane></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse><specular>0 0 0 0</specular><emissive>1 0.9 0.2 1</emissive></material></visual>";
            }
            return roadString;
        }
        std::string getRoadSDFStringWhiteLine(){
            std::string roadString = "";
            std::vector<Point> pointsToDisplay = getRoadPoints(0.1);
            for (int index = 0; index<static_cast<int>(pointsToDisplay.size());index++){
                Point point = pointsToDisplay.at(index);
                roadString+="<visual name ='visual"+ std::to_string(this->visualTagIndex)+"-"+std::to_string(index)+"'><pose>";
                roadString+= "" +std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z) + " 0 0 " +std::to_string(point.rotation);
                roadString+="</pose><geometry><plane><normal>0 0 1</normal> <size>0.1 0.03</size></plane></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse><specular>0 0 0 0</specular><emissive>1 1 1 1</emissive></material></visual>";
            }
            return roadString;
        }
        std::string getRoadSDFStringDottedYellowLine(){
            std::string roadString = "";
            std::vector<Point> pointsToDisplay = getRoadPoints(0.3);
            for (int index = 0; index<static_cast<int>(pointsToDisplay.size());index++){
                Point point = pointsToDisplay.at(index);
                roadString+="<visual name ='visual"+ std::to_string(this->visualTagIndex)+"-"+std::to_string(index)+"'><pose>";
                roadString+= "" +std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z) + " 0 0 " +std::to_string(point.rotation);
                roadString+="</pose><geometry><plane><normal>0 0 1</normal> <size>0.1 0.03</size></plane></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse><specular>0 0 0 0</specular><emissive>1 0.9 0.2 1</emissive></material></visual>";
            }
            return roadString;
        }
        std::string getRoadSDFStringDottedWhiteLine(){
            std::string roadString = "";
            std::vector<Point> pointsToDisplay = getRoadPoints(0.3);
            for (int index = 0; index<static_cast<int>(pointsToDisplay.size());index++){
                Point point = pointsToDisplay.at(index);
                roadString+="<visual name ='visual"+ std::to_string(this->visualTagIndex)+"-"+std::to_string(index)+"'><pose>";
                roadString+= "" +std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z) + " 0 0 " +std::to_string(point.rotation);
                roadString+="</pose><geometry><plane><normal>0 0 1</normal> <size>0.1 0.03</size></plane></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse><specular>0 0 0 0</specular><emissive>1 1 1 1</emissive></material></visual>";
            }

            return roadString;
        }
    private:
        double distanceBetweenPoints(Point first,Point second){
          return  sqrt(pow((first.x-second.x),2) + pow((first.y-second.y),2));
        }

};
class Road{
    public:
    std::vector<RoadLine> roadLines;
    Road(std::vector<RoadLine> lines){
        roadLines = lines;
    }
    Road(int trackNumber){
        if(trackNumber == 1){
            RoadSegment line1(0,0,0,0,5,M_PI/2,0);
            RoadSegment circle1(2,5,0,1,2,M_PI,0);
            RoadSegment line2(4,5,0,0,5,-M_PI/2,0);
            RoadSegment circle2(2,0,0,1,2,0,-M_PI);
            std::vector<RoadSegment> yellowSegments;
            yellowSegments.push_back(line1);
            yellowSegments.push_back(circle1);
            yellowSegments.push_back(line2);
            yellowSegments.push_back(circle2);
            RoadLine yellow(yellowSegments,false,true,0);

            RoadSegment line3(0-0.4,0,0,0,5,M_PI/2,0);
            RoadSegment circle3(2,5,0,1,2+0.4,M_PI,0);
            RoadSegment line4(4+0.4,5,0,0,5,-M_PI/2,0);
            RoadSegment circle4(2,0,0,1,2+0.4,0,-M_PI);
            std::vector<RoadSegment> innerWhiteSegments;
            innerWhiteSegments.push_back(line3);
            innerWhiteSegments.push_back(circle3);
            innerWhiteSegments.push_back(line4);
            innerWhiteSegments.push_back(circle4);
            RoadLine innerWhite(innerWhiteSegments,1);

            RoadSegment line5(0+0.4,0,0,0,5,M_PI/2,0);
            RoadSegment circle5(2,5,0,1,2-0.4,M_PI,0);
            RoadSegment line6(4-0.4,5,0,0,5,-M_PI/2,0);
            RoadSegment circle6(2,0,0,1,2-0.4,0,-M_PI);
            std::vector<RoadSegment> outerWhiteSegments;
            outerWhiteSegments.push_back(line5);
            outerWhiteSegments.push_back(circle5);
            outerWhiteSegments.push_back(line6);
            outerWhiteSegments.push_back(circle6);
            RoadLine outerWhite(outerWhiteSegments,2);

            std::vector<RoadLine> roadLines;
            roadLines.push_back(yellow);
            roadLines.push_back(innerWhite);
            roadLines.push_back(outerWhite);
            this->roadLines = roadLines;
        }
        if(trackNumber == 2){
            RoadSegment line1(-1,0,0,0,2,0,0);
            RoadSegment line2(5,4,0,0,4,0,0);
            RoadSegment line3(9,12,0,0,4,M_PI,0);
            RoadSegment line4(1,16,0,0,2,M_PI,0);
            RoadSegment line5(-5,12,0,0,4,M_PI,0);
            RoadSegment line6(-9,4,0,0,4,0,0);
            RoadSegment circle1(1,2,0,1,2,M_PI*3/2,M_PI*2);
            RoadSegment circle2(5,2,0,1,2,M_PI,M_PI_2);
            RoadSegment circle3(9,8,0,1,4,M_PI*3/2,M_PI*2);
            RoadSegment circle4(9,8,0,1,4,0,M_PI_2);
            RoadSegment circle5(5,14,0,1,2,M_PI*3/2,M_PI);
            RoadSegment circle6(1,14,0,1,2,0,M_PI_2);
            RoadSegment circle7(-1,14,0,1,2,M_PI_2,M_PI);
            RoadSegment circle8(-5,14,0,1,2,2*M_PI,M_PI*3/2);
            RoadSegment circle9(-9,8,0,1,4,M_PI_2,M_PI);
            RoadSegment circle10(-9,8,0,1,4,M_PI,M_PI*3/2);
            RoadSegment circle11(-5,2,0,1,2,M_PI_2,0);
            RoadSegment circle12(-1,2,0,1,2,M_PI,M_PI*3/2);
            std::vector<RoadSegment> yellowSegments;
            yellowSegments.push_back(line1);
            yellowSegments.push_back(circle1);
            yellowSegments.push_back(circle2);
            yellowSegments.push_back(line2);
            yellowSegments.push_back(circle3);
            yellowSegments.push_back(circle4);
            yellowSegments.push_back(line3);
            yellowSegments.push_back(circle5);
            yellowSegments.push_back(circle6);
            yellowSegments.push_back(line4);
            yellowSegments.push_back(circle7);
            yellowSegments.push_back(circle8);
            yellowSegments.push_back(line5);
            yellowSegments.push_back(circle9);
            yellowSegments.push_back(circle10);
            yellowSegments.push_back(line6);
            yellowSegments.push_back(circle11);
            yellowSegments.push_back(circle12);
            RoadLine yellow(yellowSegments,false,true,0);

            RoadSegment w1line1(-1,0+0.4,0,0,2,0,0);
            RoadSegment w1line2(5,4+0.4,0,0,4,0,0);
            RoadSegment w1line3(9,12-0.4,0,0,4,M_PI,0);
            RoadSegment w1line4(1,16-0.4,0,0,2,M_PI,0);
            RoadSegment w1line5(-5,12-0.4,0,0,4,M_PI,0);
            RoadSegment w1line6(-9,4+0.4,0,0,4,0,0);
            RoadSegment w1circle1(1,2,0,1,2-0.4,M_PI*3/2,M_PI*2);
            RoadSegment w1circle2(5,2,0,1,2+0.4,M_PI,M_PI_2);
            RoadSegment w1circle3(9,8,0,1,4-0.4,M_PI*3/2,M_PI*2);
            RoadSegment w1circle4(9,8,0,1,4-0.4,0,M_PI_2);
            RoadSegment w1circle5(5,14,0,1,2+0.4,M_PI*3/2,M_PI);
            RoadSegment w1circle6(1,14,0,1,2-0.4,0,M_PI_2);
            RoadSegment w1circle7(-1,14,0,1,2-0.4,M_PI_2,M_PI);
            RoadSegment w1circle8(-5,14,0,1,2+0.4,2*M_PI,M_PI*3/2);
            RoadSegment w1circle9(-9,8,0,1,4-0.4,M_PI_2,M_PI);
            RoadSegment w1circle10(-9,8,0,1,4-0.4,M_PI,M_PI*3/2);
            RoadSegment w1circle11(-5,2,0,1,2+0.4,M_PI_2,0);
            RoadSegment w1circle12(-1,2,0,1,2-0.4,M_PI,M_PI*3/2);
            std::vector<RoadSegment> innerWhiteSegments;
            innerWhiteSegments.push_back(w1line1);
            innerWhiteSegments.push_back(w1circle1);
            innerWhiteSegments.push_back(w1circle2);
            innerWhiteSegments.push_back(w1line2);
            innerWhiteSegments.push_back(w1circle3);
            innerWhiteSegments.push_back(w1circle4);
            innerWhiteSegments.push_back(w1line3);
            innerWhiteSegments.push_back(w1circle5);
            innerWhiteSegments.push_back(w1circle6);
            innerWhiteSegments.push_back(w1line4);
            innerWhiteSegments.push_back(w1circle7);
            innerWhiteSegments.push_back(w1circle8);
            innerWhiteSegments.push_back(w1line5);
            innerWhiteSegments.push_back(w1circle9);
            innerWhiteSegments.push_back(w1circle10);
            innerWhiteSegments.push_back(w1line6);
            innerWhiteSegments.push_back(w1circle11);
            innerWhiteSegments.push_back(w1circle12);
            RoadLine innerWhite(innerWhiteSegments,1);

            RoadSegment w2line1(-1,0-0.4,0,0,2,0,0);
            RoadSegment w2line2(5,4-0.4,0,0,4,0,0);
            RoadSegment w2line3(9,12+0.4,0,0,4,M_PI,0);
            RoadSegment w2line4(1,16+0.4,0,0,2,M_PI,0);
            RoadSegment w2line5(-5,12+0.4,0,0,4,M_PI,0);
            RoadSegment w2line6(-9,4-0.4,0,0,4,0,0);
            RoadSegment w2circle1(1,2,0,1,2+0.4,M_PI*3/2,M_PI*2);
            RoadSegment w2circle2(5,2,0,1,2-0.4,M_PI,M_PI_2);
            RoadSegment w2circle3(9,8,0,1,4+0.4,M_PI*3/2,M_PI*2);
            RoadSegment w2circle4(9,8,0,1,4+0.4,0,M_PI_2);
            RoadSegment w2circle5(5,14,0,1,2-0.4,M_PI*3/2,M_PI);
            RoadSegment w2circle6(1,14,0,1,2+0.4,0,M_PI_2);
            RoadSegment w2circle7(-1,14,0,1,2+0.4,M_PI_2,M_PI);
            RoadSegment w2circle8(-5,14,0,1,2-0.4,2*M_PI,M_PI*3/2);
            RoadSegment w2circle9(-9,8,0,1,4+0.4,M_PI_2,M_PI);
            RoadSegment w2circle10(-9,8,0,1,4+0.4,M_PI,M_PI*3/2);
            RoadSegment w2circle11(-5,2,0,1,2-0.4,M_PI_2,0);
            RoadSegment w2circle12(-1,2,0,1,2+0.4,M_PI,M_PI*3/2);
            std::vector<RoadSegment> outerWhiteSegments;
            outerWhiteSegments.push_back(w2line1);
            outerWhiteSegments.push_back(w2circle1);
            outerWhiteSegments.push_back(w2circle2);
            outerWhiteSegments.push_back(w2line2);
            outerWhiteSegments.push_back(w2circle3);
            outerWhiteSegments.push_back(w2circle4);
            outerWhiteSegments.push_back(w2line3);
            outerWhiteSegments.push_back(w2circle5);
            outerWhiteSegments.push_back(w2circle6);
            outerWhiteSegments.push_back(w2line4);
            outerWhiteSegments.push_back(w2circle7);
            outerWhiteSegments.push_back(w2circle8);
            outerWhiteSegments.push_back(w2line5);
            outerWhiteSegments.push_back(w2circle9);
            outerWhiteSegments.push_back(w2circle10);
            outerWhiteSegments.push_back(w2line6);
            outerWhiteSegments.push_back(w2circle11);
            outerWhiteSegments.push_back(w2circle12);
            RoadLine outerWhite(outerWhiteSegments,2);

            std::vector<RoadLine> roadLines;
            roadLines.push_back(yellow);
            roadLines.push_back(innerWhite);
            roadLines.push_back(outerWhite);
            this->roadLines = roadLines;
        }
    }
    std::vector<int> closestLineIndexes(Point referencePoint){
      std::vector<int> closestLines;
      std::vector<int> indexes;
      std::vector<double> distances;
      for(int i =0; i<static_cast<int>(roadLines.size());i++){
        indexes.push_back(i);
        distances.push_back(roadLines.at(i).distanceToClosestPoint(referencePoint));
      }
      while(indexes.size()>0){
        double minimumDistance = distances.at(0);
        int indexToRemoveAt = 0;
        for(int j =0; j<static_cast<int>(indexes.size());j++){
          if(distances.at(j)<minimumDistance){
            minimumDistance = distances.at(j);
            indexToRemoveAt = j;
          }
        }
        closestLines.push_back(indexes.at(indexToRemoveAt));
        indexes.erase(indexes.begin()+indexToRemoveAt);
        distances.erase(distances.begin()+indexToRemoveAt);
      }
      return closestLines;
    }
    std::vector<int> getLanesToTheLeftOfPoint(Point referencePoint){
      std::vector<int> closestLines = this->closestLineIndexes(referencePoint);
      std::vector<int> leftLanes;
      for(int i =0; i<static_cast<int>(closestLines.size());i++){
        if(this->isLaneToTheLeftOfPoint(referencePoint,closestLines.at(i))){
          leftLanes.push_back(closestLines.at(i));
        }
      }
      return leftLanes;
    }
    std::vector<int> getLanesToTheRightOfPoint(Point referencePoint){
      std::vector<int> closestLines = this->closestLineIndexes(referencePoint);
      std::vector<int> rightLanes;
      for(int i =0; i<static_cast<int>(closestLines.size());i++){
        if(!this->isLaneToTheLeftOfPoint(referencePoint,closestLines.at(i))){
          rightLanes.push_back(closestLines.at(i));
        }
      }
      return rightLanes;
    }
    bool isLaneToTheLeftOfPoint(Point referencePoint,int roadIndex){
      Point closestPointOnLine = this->roadLines.at(roadIndex).getClosestPointOnRoad(referencePoint);
      double angleToRotateBy =-1 * (referencePoint.rotation - M_PI_2);

      double pointXBeforeRotation = closestPointOnLine.x - referencePoint.x;
      double pointYBeforeRotation = closestPointOnLine.y - referencePoint.y;
      double newXForPoint = pointXBeforeRotation * cos(angleToRotateBy) - pointYBeforeRotation *sin(angleToRotateBy);
      if(newXForPoint < 0){
        return true;
      }
      return false;
    }
    std::string getRoadModel(){
        std::string roadString = "<sdf version ='1.5'><model name ='TrackLines'><static>true</static><pose>0 0 0 0 0 0</pose><link name ='link'><pose>0 0 0 0 0 0</pose>";
        for(int index=0;index<static_cast<int>(roadLines.size());index++){
            roadString+=roadLines.at(index).getRoadSDFString();
        }
        roadString+="</link></model></sdf>";
        return roadString;
    }
};
ofstream followingFile;
ofstream rightTurningFile;
ofstream leftTurningFile;
ofstream laneStateFile;
ofstream turningDirectionFile;
static const std::string OPENCV_WINDOW = "Image window";
int followingImageCount = 0;
int rightTurningImageCount = 0;
int leftTurningImageCount = 0;
int laneStateImageCount = 0;
int turningDirectionCount = 0;
bool turningLeft = false;
int recorderTurningDirection = 0;
void setFileCount(){
   std::filesystem::path followingFilePath { "/home/ryan/TrainingData/simulator_following_data/photos" };
   for (auto& p : std::filesystem::directory_iterator(followingFilePath))
   {
      ++followingImageCount;
   }
   std::filesystem::path rightTurningFilePath { "/home/ryan/TrainingData/simulator_right_lane_switch/photos" };
   for (auto& p : std::filesystem::directory_iterator(rightTurningFilePath))
   {
      ++rightTurningImageCount;
   }
   std::filesystem::path leftTurningFilePath { "/home/ryan/TrainingData/simulator_left_lane_switch/photos" };
   for (auto& p : std::filesystem::directory_iterator(leftTurningFilePath))
   {
      ++leftTurningImageCount;
   }
   std::filesystem::path laneStateFilePath { "/home/ryan/TrainingData/simulator_lane_state/photos" };
   for (auto& p : std::filesystem::directory_iterator(laneStateFilePath))
   {
      ++laneStateImageCount;
   }
   std::filesystem::path turningDirectionFilePath { "/home/ryan/TrainingData/simulator_turning_direction/photos" };
   for (auto& p : std::filesystem::directory_iterator(turningDirectionFilePath))
   {
      ++turningDirectionCount;
   }
 }
class OvertakingRecorder{
  private:
    ros::NodeHandle n;
    ros::Publisher drive_pub;
    ros::Subscriber model_sub;
    ros::Subscriber scan_sub;
    ros::Subscriber controls_sub;
    image_transport::ImageTransport image_transport;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    std::string carName = "";
    double speed = 0;
    Road road = Road(2);
    std::vector<int> overtakingLane;
    std::vector<int> previousLane;
    std::vector<float> filteredLidarReadings;
    sensor_msgs::LaserScanConstPtr lidar;
    cv::Mat currentSemanticSegmentationImage;
    long timeOfLastCall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // States: 
    // 0 = normal driving
    // 1 = merging into side lane 
    // 2 = in side lane
    // 3 = merging back into normal lane
    int drivingState = 0; 
    long actionTimer = 0;
    bool canTurnBackIntoLane = false;
    double lastRecordedSteeringAngle = 0;
    double lastRecordedSpeed = -1;
    bool worldRunning = false;
    double lastLaneState = 0;
  public:
    OvertakingRecorder():image_transport(n){
        n = ros::NodeHandle("~");
        std::string drive_topic, model_topic, scan_topic;
        n.getParam("nav_drive_topic", drive_topic);
        n.getParam("scan_topic", scan_topic);
        n.getParam("/model_topic",model_topic);
        n.getParam("car_name",carName);
        n.getParam("speed",speed);
        
        drive_pub = n.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);
        model_sub = n.subscribe(model_topic, 1, &OvertakingRecorder::model_state_callback,this);
        scan_sub = n.subscribe(scan_topic, 1, &OvertakingRecorder::scan_callback,this);
        image_sub_ = image_transport.subscribe("/semantic_segmentation/car_output", 1, &OvertakingRecorder::image_callback, this);
        controls_sub = n.subscribe("/simulator/command/input",1000,&OvertakingRecorder::command_callback,this);
        cv::namedWindow(OPENCV_WINDOW);
    }
    ~OvertakingRecorder()
    {
      cv::destroyWindow(OPENCV_WINDOW);
      leftTurningFile.close();
      rightTurningFile.close();
      followingFile.close();
      laneStateFile.close();
      turningDirectionFile.close();
    }
    int laneState(Point car,Point firstPoint,Point secondPoint){
      // 00 out of lane to 
      double angleToRotateBy =-1 * (car.rotation - M_PI_2);

      double pointXBeforeRotationFirst = firstPoint.x - car.x;
      double pointYBeforeRotationFirst = firstPoint.y - car.y;
      double newXForPointFirst = pointXBeforeRotationFirst * cos(angleToRotateBy) - pointYBeforeRotationFirst *sin(angleToRotateBy);
      double newYForPointFirst = pointYBeforeRotationFirst * cos(angleToRotateBy) + pointXBeforeRotationFirst *sin(angleToRotateBy);

      double pointXBeforeRotationSecond = secondPoint.x - car.x;
      double pointYBeforeRotationSecond = secondPoint.y - car.y;
      double newXForPointSecond = pointXBeforeRotationSecond * cos(angleToRotateBy) - pointYBeforeRotationSecond *sin(angleToRotateBy);
      double newYForPointSecond = pointYBeforeRotationSecond * cos(angleToRotateBy) + pointXBeforeRotationSecond *sin(angleToRotateBy);

      double distanceFromCenter = ((newXForPointFirst + newXForPointSecond) / 2);
      double distanceBetweenPoints = abs(newXForPointFirst) + abs(newXForPointSecond);
      if((abs(distanceFromCenter) / distanceBetweenPoints) < 0.2){
        if(distanceFromCenter > 0){
          return 3;

        } else{
          return 4;
        }
      }
      else {
        if(distanceFromCenter > 0){
          return 1;

        } else{
          return 2;
        }
      }
      return 0;
    }
    bool isCarInLane(Point car,Point firstPoint,Point secondPoint){
      double angleToRotateBy =-1 * (car.rotation - M_PI_2);

      double pointXBeforeRotationFirst = firstPoint.x - car.x;
      double pointYBeforeRotationFirst = firstPoint.y - car.y;
      double newXForPointFirst = pointXBeforeRotationFirst * cos(angleToRotateBy) - pointYBeforeRotationFirst *sin(angleToRotateBy);
      double newYForPointFirst = pointYBeforeRotationFirst * cos(angleToRotateBy) + pointXBeforeRotationFirst *sin(angleToRotateBy);

      double pointXBeforeRotationSecond = secondPoint.x - car.x;
      double pointYBeforeRotationSecond = secondPoint.y - car.y;
      double newXForPointSecond = pointXBeforeRotationSecond * cos(angleToRotateBy) - pointYBeforeRotationSecond *sin(angleToRotateBy);
      double newYForPointSecond = pointYBeforeRotationSecond * cos(angleToRotateBy) + pointXBeforeRotationSecond *sin(angleToRotateBy);

      double distanceFromCenter = abs((newXForPointFirst + newXForPointSecond) / 2);
      double distanceBetweenPoints = abs(newXForPointFirst) + abs(newXForPointSecond);
      return (distanceFromCenter / distanceBetweenPoints) < 0.2; // find good ratio
    }
    bool isItSafeToTurn(Point car, Point firstPoint,Point secondPoint){
      double safeVerticalDistanceInLane = 1;
      double heightAdjustment = safeVerticalDistanceInLane/2;
      Point middlePointOfLane;
      middlePointOfLane.rotation = firstPoint.rotation + secondPoint.rotation /2;
      middlePointOfLane.x = (firstPoint.x + secondPoint.x) /2;
      middlePointOfLane.y = (firstPoint.y + secondPoint.y) /2;
      middlePointOfLane.z = (firstPoint.z + secondPoint.z) /2;
      //transform middle position to car coordinate frame
      double angleToRotateBy =-1 * (car.rotation - M_PI_2);
      double pointXBeforeRotation = middlePointOfLane.x - car.x;
      double pointYBeforeRotation = middlePointOfLane.y - car.y;
      double newXForPoint = pointXBeforeRotation * cos(angleToRotateBy) - pointYBeforeRotation *sin(angleToRotateBy);
      double newYForPoint = pointYBeforeRotation * cos(angleToRotateBy) + pointXBeforeRotation *sin(angleToRotateBy);
      double hypotenuse = sqrt(pow(newXForPoint,2)+pow(newYForPoint,2));
      //find angle of middle point in car frame
      double angleOfLaneMiddlePointInCarFrame = acos(newXForPoint/hypotenuse);
      //find how far the lidar should sweep on both sides to check if a car is there
      double angleToConsiderRanges = atan(heightAdjustment/hypotenuse);
      // subtract M_PI_2 to change standard coordinate plane to lidar frame where angle of zero is in the top direction
      double angleUpperBound = angleOfLaneMiddlePointInCarFrame + angleToConsiderRanges - M_PI_2;
      double angleLowerBound = angleOfLaneMiddlePointInCarFrame - angleToConsiderRanges - M_PI_2;
      // find what range is safe in that sweeping area
      double safeSweepingRange = sqrt(pow(hypotenuse,2)+pow(heightAdjustment,2));
      //loop through lidar to see if safe
      bool isSafeToTurn = true;
      
      for(int i = 0; i < static_cast<int>(filteredLidarReadings.size());i++){
        double angleChecking = lidar->angle_min + lidar->angle_increment * i;
        if(angleChecking < angleUpperBound && angleChecking > angleLowerBound){
          if(filteredLidarReadings.at(i)<safeSweepingRange){
            isSafeToTurn = false;
          }
        }
      }
      return isSafeToTurn;

    }
    double linear_distance_stearing_adjustments(Point car, Point firstPoint,Point secondPoint){

      double angleToRotateBy =-1 * (car.rotation - M_PI_2);

      double pointXBeforeRotationFirst = firstPoint.x - car.x;
      double pointYBeforeRotationFirst = firstPoint.y - car.y;
      double newXForPointFirst = pointXBeforeRotationFirst * cos(angleToRotateBy) - pointYBeforeRotationFirst *sin(angleToRotateBy);
      double newYForPointFirst = pointYBeforeRotationFirst * cos(angleToRotateBy) + pointXBeforeRotationFirst *sin(angleToRotateBy);

      double pointXBeforeRotationSecond = secondPoint.x - car.x;
      double pointYBeforeRotationSecond = secondPoint.y - car.y;
      double newXForPointSecond = pointXBeforeRotationSecond * cos(angleToRotateBy) - pointYBeforeRotationSecond *sin(angleToRotateBy);
      double newYForPointSecond = pointYBeforeRotationSecond * cos(angleToRotateBy) + pointXBeforeRotationSecond *sin(angleToRotateBy);

      double averageDistance = (newXForPointFirst + newXForPointSecond) / 2;
      int sign = 1;
      if(averageDistance > 0){
        sign = -1;
      }
      double adjustmentConstant = 1.6; //was 0.8
      double adjustment =abs(averageDistance) * adjustmentConstant;
      if(adjustment>.4){
        adjustment=.4;
      }
      adjustment*=sign;
      return adjustment;
    }
    double roadCurve(Point car, Point firstRoadLine,Point secondRoadLine){
      double angleToAdjustToCurvature = 0;
      double adjustedAngleForRoadLines = (firstRoadLine.rotation + secondRoadLine.rotation) /2;
      if(abs(firstRoadLine.rotation - secondRoadLine.rotation)>M_PI){
        adjustedAngleForRoadLines = firstRoadLine.rotation;
      }
      if(abs(car.rotation-adjustedAngleForRoadLines)<M_PI){
          angleToAdjustToCurvature = adjustedAngleForRoadLines - car.rotation;
      }else{
        if(car.rotation<adjustedAngleForRoadLines){
          angleToAdjustToCurvature = adjustedAngleForRoadLines - car.rotation - M_PI * 2;
        }else{
          angleToAdjustToCurvature = adjustedAngleForRoadLines - car.rotation + M_PI * 2;
        }
      }
      return angleToAdjustToCurvature *1.4; //1.2 tries to account for future curvature
    }
    double steering_angle_from_points(Point car, Point firstRoadLine,Point secondRoadLine){
      //For Curve adjustments
      double angleToAdjustToCurvature = 0;
      //To Keep Car in Center of Lane
      double angleToAdjustForCentering = 0;
      
      // Curve adjustments
      while(car.rotation  <0){
        car.rotation  += M_PI *2;
      }
      while(car.rotation  >M_PI *2){
        car.rotation  -= M_PI *2;
      }
      while(firstRoadLine.rotation  <0){
        firstRoadLine.rotation  += M_PI *2;
      }
      while(firstRoadLine.rotation  >M_PI *2){
        firstRoadLine.rotation  -= M_PI *2;
      }
      while(secondRoadLine.rotation <0){
        secondRoadLine.rotation += M_PI *2;
      }
      while(secondRoadLine.rotation >M_PI *2){
        secondRoadLine.rotation -= M_PI *2;
      }
      angleToAdjustToCurvature = roadCurve(car,firstRoadLine,secondRoadLine);
      Point reverseCarDirection;
      reverseCarDirection.rotation = car.rotation + M_PI;
      if(reverseCarDirection.rotation > M_PI*2){
        reverseCarDirection.rotation-= M_PI*2;
      }
      reverseCarDirection.x = car.x;
      reverseCarDirection.y = car.y;
      reverseCarDirection.z = car.z;
      double curveInOppositeDirection = roadCurve(reverseCarDirection,firstRoadLine,secondRoadLine);
      if(abs(curveInOppositeDirection)<abs(angleToAdjustToCurvature)){
        angleToAdjustToCurvature = curveInOppositeDirection;
      }
      //Lane Line Adjustments
      angleToAdjustForCentering = linear_distance_stearing_adjustments(car,firstRoadLine,secondRoadLine);
      if(abs(angleToAdjustForCentering)>0.8){
        if(angleToAdjustForCentering >0){
          angleToAdjustForCentering = 0.8;
        }else{
          angleToAdjustForCentering = -0.8;
        }
      }
      return angleToAdjustToCurvature + angleToAdjustForCentering;
    } 
    void setSwitchLaneInformation(Point currentCarLocation){
      //eventually this will need to change for differnt road structures
      //not guarenteed to follow disjoint tracks / lines
      std::vector<int> leftLines = this->road.getLanesToTheLeftOfPoint(currentCarLocation);
      std::vector<int> rightLines = this->road.getLanesToTheRightOfPoint(currentCarLocation);
      std::vector<int> lines = this->road.closestLineIndexes(currentCarLocation);

      if(static_cast<int>(leftLines.size())>=2){
        overtakingLane.clear();
        overtakingLane.push_back(leftLines.at(0));
        overtakingLane.push_back(leftLines.at(1));
        previousLane.clear();
        previousLane.push_back(lines.at(0));
        previousLane.push_back(lines.at(1));
        turningLeft = true;
      }
      else if(static_cast<int>(rightLines.size())>=2){
        overtakingLane.clear();
        overtakingLane.push_back(rightLines.at(0));
        overtakingLane.push_back(rightLines.at(1));
        previousLane.clear();
        previousLane.push_back(lines.at(0));
        previousLane.push_back(lines.at(1));
        turningLeft = false;
      }else{
        drivingState = 0;
      }
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
        return sqrt(2*1.7*std::max(0.0,minDistance-0.5)); // 1.7 is max acccleration of the car
      }
      return 0;
    }
    void followRoad(Point currentCarLocation){
      double desiredSteeringAngle = 0;
      double desiredSpeed = this->speed;

      std::vector<int> lines = this->road.closestLineIndexes(currentCarLocation);
      Point firstPointForLaneState = this->road.roadLines.at(lines.at(0)).getClosestPointOnRoad(currentCarLocation);
      Point secondPointForLaneState = this->road.roadLines.at(lines.at(1)).getClosestPointOnRoad(currentCarLocation);
      int lanePosition = laneState(currentCarLocation,firstPointForLaneState,secondPointForLaneState);
      lastLaneState = lanePosition;
      //ROS_INFO("Lane Position State: %s",std::to_string(lanePosition).c_str());

      if(drivingState == 0){
        std::vector<int> closestLines = this->road.closestLineIndexes(currentCarLocation);
        Point closestPoint = this->road.roadLines.at(closestLines.at(0)).getClosestPointOnRoad(currentCarLocation);
        Point secondClosestPoint = this->road.roadLines.at(closestLines.at(1)).getClosestPointOnRoad(currentCarLocation);
        desiredSteeringAngle = steering_angle_from_points(currentCarLocation,closestPoint,secondClosestPoint);
        desiredSpeed = safeSpeedToFollowAt();
        long currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if(desiredSpeed >=1.5){
          actionTimer = 0;
        }
        if(desiredSpeed<1.5){
          std::vector<int> leftLines = this->road.getLanesToTheLeftOfPoint(currentCarLocation);
          std::vector<int> rightLines = this->road.getLanesToTheRightOfPoint(currentCarLocation);
          std::vector<int> lines = this->road.closestLineIndexes(currentCarLocation);
          if(static_cast<int>(leftLines.size())>=2){
            recorderTurningDirection = 1;
          }
          else if(static_cast<int>(rightLines.size())>=2){
            recorderTurningDirection = 2;
          }else{
            recorderTurningDirection = 0;
          }

          actionTimer += currentTime -timeOfLastCall;
          if(actionTimer>5000){
            actionTimer=0;
            drivingState = 1;
            setSwitchLaneInformation(currentCarLocation);
          }
        }
      }

      if(drivingState == 1){
        Point closestPoint = this->road.roadLines.at(overtakingLane.at(0)).getClosestPointOnRoad(currentCarLocation);
        Point secondClosestPoint = this->road.roadLines.at(overtakingLane.at(1)).getClosestPointOnRoad(currentCarLocation);
        desiredSteeringAngle = steering_angle_from_points(currentCarLocation,closestPoint,secondClosestPoint);
        desiredSpeed = 4;
        if(isCarInLane(currentCarLocation,closestPoint,secondClosestPoint)){
          drivingState = 2;
        }
      }

      if(drivingState == 2){
        Point closestPoint = this->road.roadLines.at(overtakingLane.at(0)).getClosestPointOnRoad(currentCarLocation);
        Point secondClosestPoint = this->road.roadLines.at(overtakingLane.at(1)).getClosestPointOnRoad(currentCarLocation);
        desiredSteeringAngle = steering_angle_from_points(currentCarLocation,closestPoint,secondClosestPoint);
        desiredSpeed = 4;
        Point closestPointInOtherLane = this->road.roadLines.at(previousLane.at(0)).getClosestPointOnRoad(currentCarLocation);
        Point secondClosestPointInOtherLane = this->road.roadLines.at(previousLane.at(1)).getClosestPointOnRoad(currentCarLocation);
        if(isItSafeToTurn(currentCarLocation,closestPointInOtherLane,secondClosestPointInOtherLane)){
          if(canTurnBackIntoLane){
            long currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            actionTimer += currentTime -timeOfLastCall;
            if(actionTimer>500){
              actionTimer=0;
              drivingState = 3;
              canTurnBackIntoLane = false;
            }
          }
        }else{
          canTurnBackIntoLane = true;
        }
      }

      if(drivingState == 3){
        Point closestPoint = this->road.roadLines.at(previousLane.at(0)).getClosestPointOnRoad(currentCarLocation);
        Point secondClosestPoint = this->road.roadLines.at(previousLane.at(1)).getClosestPointOnRoad(currentCarLocation);
        desiredSteeringAngle = steering_angle_from_points(currentCarLocation,closestPoint,secondClosestPoint);
        desiredSpeed = 4;
        if(isCarInLane(currentCarLocation,closestPoint,secondClosestPoint)){
          drivingState = 0;
        }
      }
      ackermann_msgs::AckermannDriveStamped drive_st_msg;
      ackermann_msgs::AckermannDrive drive_msg;
      drive_msg.steering_angle = desiredSteeringAngle;
      drive_msg.speed = desiredSpeed;
      drive_st_msg.drive = drive_msg;
      drive_pub.publish(drive_st_msg);
      timeOfLastCall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      lastRecordedSteeringAngle = drive_msg.steering_angle;
      lastRecordedSpeed = drive_msg.speed;
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
    void model_state_callback(const gazebo_msgs::ModelStatesConstPtr& msg){
      if(msg->pose.size()>0){
        for(int i = 0; i < static_cast<int>(msg->name.size());i++){
          if(msg->name.at(i)==carName){
            geometry_msgs::Quaternion orientation =  msg->pose[i].orientation;
            tf2::Quaternion quaternion(orientation.x,orientation.y,orientation.z,orientation.w);
            tf2::Matrix3x3 m(quaternion);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            geometry_msgs::Point position=  msg->pose[i].position;
            Point currentCarLocation;
            currentCarLocation.rotation = yaw;
            currentCarLocation.x = position.x;
            currentCarLocation.y = position.y;
            currentCarLocation.z = position.z;
            followRoad(currentCarLocation);
          }
        }
      }
    }
    void command_callback(const std_msgs::Int64ConstPtr &msg){
        if(((int)msg->data) == 32){
            if(worldRunning){
              ROS_INFO("World Paused: IOSTREAM CLOSED");
              worldRunning = false;
              leftTurningFile.close();
              rightTurningFile.close();
              followingFile.close();
              laneStateFile.close();
              turningDirectionFile.close();

              leftTurningFile.clear();
              rightTurningFile.clear();
              followingFile.clear();
              laneStateFile.clear();
              turningDirectionFile.clear();

              lastRecordedSpeed = -1;
            }else{
              ROS_INFO("World Resumed: IOSTREAM OPENED");
              if(!leftTurningFile.is_open()){
                leftTurningFile.open("/home/ryan/TrainingData/simulator_left_lane_switch/turningData.bin", ios::out | ios::app | ios::binary);
              }
              if(!rightTurningFile.is_open()){
                rightTurningFile.open("/home/ryan/TrainingData/simulator_right_lane_switch/turningData.bin", ios::out | ios::app | ios::binary);
              }
              if(!followingFile.is_open()){
                followingFile.open("/home/ryan/TrainingData/simulator_following_data/followingData.bin", ios::out | ios::app | ios::binary);
              }
              if(!laneStateFile.is_open()){
                laneStateFile.open("/home/ryan/TrainingData/simulator_lane_state/laneState.bin",ios::out | ios::app | ios::binary);
              }
              if(!turningDirectionFile.is_open()){
                turningDirectionFile.open("/home/ryan/TrainingData/simulator_turning_direction/turningDirection.bin",ios::out | ios::app | ios::binary);
              }
              worldRunning = true;
            }
        }
    }
    void image_callback(const sensor_msgs::ImageConstPtr& msg){
      try
      {
        // for color camera
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
        cv::resize(cv_ptr->image,currentSemanticSegmentationImage,cv::Size(256,256));
        cv::imshow(OPENCV_WINDOW, currentSemanticSegmentationImage);
        cv::waitKey(3);
        if(lastRecordedSpeed >= 0 && worldRunning){
          if(drivingState == 0){
            followingFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
            followingFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
            cv::imwrite(("/home/ryan/TrainingData/simulator_following_data/photos/"+to_string(followingImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            followingImageCount += 1;

            // //record if it is in lane line
            // laneStateFile.write(reinterpret_cast<char*>( & lastLaneState ), sizeof(lastLaneState) );
            // cv::imwrite(("/home/ryan/TrainingData/simulator_lane_state/photos/"+to_string(laneStateImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            // laneStateImageCount += 1;
            // ROS_INFO("Lane Position State: %s",std::to_string(lastLaneState).c_str());

            if(actionTimer>0){
              //record Turning Direction
              turningDirectionFile.write(reinterpret_cast<char*>( &recorderTurningDirection ), sizeof( recorderTurningDirection ) );
              cv::imwrite(("/home/ryan/TrainingData/simulator_turning_direction/photos/"+to_string(turningDirectionCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
              turningDirectionCount += 1;
              ROS_INFO("Turning Direciton: %s",std::to_string(recorderTurningDirection).c_str());
            }
            
          }
          if(drivingState == 1){
            //if Turning Left is true, Record to left turning node
            if(turningLeft){
              leftTurningFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
              leftTurningFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
              cv::imwrite(("/home/ryan/TrainingData/simulator_left_lane_switch/photos/"+to_string(leftTurningImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
              leftTurningImageCount += 1;
            }else{
              rightTurningFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
              rightTurningFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
              cv::imwrite(("/home/ryan/TrainingData/simulator_right_lane_switch/photos/"+to_string(rightTurningImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
              rightTurningImageCount += 1;
            }
            //record if it is in lane line
            laneStateFile.write(reinterpret_cast<char*>( & lastLaneState ), sizeof(lastLaneState) );
            cv::imwrite(("/home/ryan/TrainingData/simulator_lane_state/photos/"+to_string(laneStateImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            laneStateImageCount += 1;
            //ROS_INFO("Lane Position State: %s",std::to_string(lastLaneState).c_str());
          }
          if(drivingState == 2){
            followingFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
            followingFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
            cv::imwrite(("/home/ryan/TrainingData/simulator_following_data/photos/"+to_string(followingImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            followingImageCount += 1;
            //record if it is in lane line
            laneStateFile.write(reinterpret_cast<char*>( & lastLaneState ), sizeof(lastLaneState) );
            cv::imwrite(("/home/ryan/TrainingData/simulator_lane_state/photos/"+to_string(laneStateImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            laneStateImageCount += 1;
            //ROS_INFO("Lane Position State: %s",std::to_string(lastLaneState).c_str());
          }
          if(drivingState == 3){
            //if Turning Left true, Record to right turning node
            //^ It will be turning back into the other lane at this state so it will be reversed
            if(!turningLeft){
              leftTurningFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
              leftTurningFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
              cv::imwrite(("/home/ryan/TrainingData/simulator_left_lane_switch/photos/"+to_string(leftTurningImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
              leftTurningImageCount += 1;
            }else{
              rightTurningFile.write(reinterpret_cast<char*>( & lastRecordedSteeringAngle ), sizeof(lastRecordedSteeringAngle) );
              rightTurningFile.write(reinterpret_cast<char*>( & lastRecordedSpeed ), sizeof(lastRecordedSpeed) );
              cv::imwrite(("/home/ryan/TrainingData/simulator_right_lane_switch/photos/"+to_string(rightTurningImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
              rightTurningImageCount += 1;
            }
            //record if it is in lane line
            laneStateFile.write(reinterpret_cast<char*>( & lastLaneState ), sizeof(lastLaneState) );
            cv::imwrite(("/home/ryan/TrainingData/simulator_lane_state/photos/"+to_string(laneStateImageCount) + "-semanticOutput.png"),currentSemanticSegmentationImage);
            laneStateImageCount += 1;
            //ROS_INFO("Lane Position State: %s",std::to_string(lastLaneState).c_str());

          }
          //assuming car speed is 0
          //myFile.write(reinterpret_cast<char*>( & carSpeed ), sizeof(carSpeed) );
          //cv::imwrite(("/home/ryan/TrainingData/speedPhotos/"+to_string(imageCount) + "-rgbImage.jpg"),imageInRGB);
        }
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
  }
};
int main(int argc, char ** argv) {
    setFileCount();
    ros::init(argc, argv, "overtaking_node", ros::init_options::AnonymousName);
    OvertakingRecorder rw;
    ros::spin();
    return 0;
}
