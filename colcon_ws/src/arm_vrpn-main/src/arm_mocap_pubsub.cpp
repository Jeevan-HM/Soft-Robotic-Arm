#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdlib.h>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <zmq.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// #include <Eigen/Dense>

using namespace std::chrono_literals;

using std::placeholders::_1;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

float x1_mocap,y1_mocap,z1_mocap;
float x2_mocap,y2_mocap,z2_mocap;
float x3_mocap,y3_mocap,z3_mocap;
float q1_mocap_x,q1_mocap_y,q1_mocap_z,q1_mocap_w;
float q2_mocap_x,q2_mocap_y,q2_mocap_z,q2_mocap_w;
float q3_mocap_x,q3_mocap_y,q3_mocap_z,q3_mocap_w;

zmq::context_t ctx;
zmq::socket_t publisher(ctx, zmq::socket_type::pub);


    // // zmq::socket_t pub(ctx, ZMQ_PUB);
    // 
    // publisher.send(zmq::str_buffer("Hello"));

class Mocapzmq_pub : public rclcpp::Node
{
  public:
    Mocapzmq_pub()
    : Node("mocap_zmq_pub"), count_(0)
    {

     auto qos_profile = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
     qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
     qos_profile.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
     qos_profile.keep_last(10);

        // vrpn subscriptions with best effort reliability
     vrpn_subscription1_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/vrpn_mocap/Rigid_Body_001/pose", qos_profile, std::bind(&Mocapzmq_pub::topic_callback1, this, _1));
     vrpn_subscription2_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/vrpn_mocap/Rigid_Body_002/pose", qos_profile, std::bind(&Mocapzmq_pub::topic_callback2, this, _1));
     vrpn_subscription3_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/vrpn_mocap/Rigid_Body_003/pose", qos_profile, std::bind(&Mocapzmq_pub::topic_callback3, this, _1));
// zmq setup

      publisher.bind("tcp://127.0.0.1:3885");
      // publisher.send(zmq::str_buffer("Hello"));
    }


  private:
    void topic_callback1(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {

      x1_mocap = msg->pose.position.x;
      y1_mocap = msg->pose.position.y;
      z1_mocap = msg->pose.position.z;
      
      // storing quaternion values
      q1_mocap_x = msg->pose.orientation.x;
      q1_mocap_y = msg->pose.orientation.y;
      q1_mocap_z = msg->pose.orientation.z;
      q1_mocap_w = msg->pose.orientation.w;
      float data[21] = {x1_mocap,y1_mocap,z1_mocap,
                      q1_mocap_x,q1_mocap_y,q1_mocap_z,q1_mocap_w,
                      x2_mocap,y2_mocap,z2_mocap,
                      q2_mocap_x,q2_mocap_y,q2_mocap_z,q2_mocap_w,
                      x3_mocap,y3_mocap,z3_mocap,
                      q3_mocap_x,q3_mocap_y,q3_mocap_z,q3_mocap_w
                    };          
      std::string str_sub_msg_1 = std::to_string(x1_mocap)+","+std::to_string(y1_mocap)+","+std::to_string(z1_mocap)+","+std::to_string(q1_mocap_x)+","+std::to_string(q1_mocap_y)+","+std::to_string(q1_mocap_z)+","+std::to_string(q1_mocap_w);
      std::string str_sub_msg_2 = std::to_string(x2_mocap)+","+std::to_string(y2_mocap)+","+std::to_string(z2_mocap)+","+std::to_string(q2_mocap_x)+","+std::to_string(q2_mocap_y)+","+std::to_string(q2_mocap_z)+","+std::to_string(q2_mocap_w);
      std::string str_sub_msg_3 = std::to_string(x3_mocap)+","+std::to_string(y3_mocap)+","+std::to_string(z3_mocap)+","+std::to_string(q3_mocap_x)+","+std::to_string(q3_mocap_y)+","+std::to_string(q3_mocap_z)+","+std::to_string(q3_mocap_w);
      std::string str_msg = str_sub_msg_1+","+str_sub_msg_2+","+str_sub_msg_3;
      zmq::message_t message(str_msg.size());
      std::memcpy(message.data(),str_msg.data(),str_msg.size()); 
      publisher.send(message);
      // std::cout<< str_msg <<"\n";
    }

    void topic_callback2(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {


      x2_mocap = msg->pose.position.x;
      y2_mocap = msg->pose.position.y;
      z2_mocap = msg->pose.position.z;
      
      // storing quaternion values
      q2_mocap_x = msg->pose.orientation.x;
      q2_mocap_y = msg->pose.orientation.y;
      q2_mocap_z = msg->pose.orientation.z;
      q2_mocap_w = msg->pose.orientation.w;
      // publisher.send(zmq::str_buffer("Hello2"));


    }

    void topic_callback3(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {


      x3_mocap = msg->pose.position.x;
      y3_mocap = msg->pose.position.y;
      z3_mocap = msg->pose.position.z;
      
      // storing quaternion values
      q3_mocap_x = msg->pose.orientation.x;
      q3_mocap_y = msg->pose.orientation.y;
      q3_mocap_z = msg->pose.orientation.z;
      q3_mocap_w = msg->pose.orientation.w;
      // publisher.send(zmq::str_buffer("Hello3")); 
    }
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vrpn_subscription1_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vrpn_subscription2_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vrpn_subscription3_;
    std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

    size_t count_;


    

};

int main(int argc, char * argv[])
{


  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Mocapzmq_pub>());
  rclcpp::shutdown();
  return 0;
}
