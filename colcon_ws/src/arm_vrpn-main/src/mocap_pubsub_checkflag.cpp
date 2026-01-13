#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/vehicle_visual_odometry.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>

using namespace std::chrono_literals;

using std::placeholders::_1;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MocapPublisher : public rclcpp::Node
{
  public:
    MocapPublisher()
    : Node("mocap_publisher"), count_(0)
    {
      px4_publisher_ = this->create_publisher<px4_msgs::msg::VehicleVisualOdometry>("VehicleVisualOdometry_PubSubTopic", 10);

      timer_ = this->create_wall_timer(
      8ms, std::bind(&MocapPublisher::timer_callback, this));

      timesync_subscription_ = this->create_subscription<px4_msgs::msg::Timesync>("Timesync_PubSubTopic", 10,
        [this](const px4_msgs::msg::Timesync::UniquePtr msg) {
          timestamp_.store(msg->timestamp);
        });

      vrpn_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/RigidBody1/pose",10,std::bind(&MocapPublisher::topic_callback, this, _1));
    }

  private:
    void timer_callback()
    {
      auto message = px4_msgs::msg::VehicleVisualOdometry();

      // message.timestamp = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();
      message.timestamp = timestamp_.load();

      // message.timestamp_sample = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();
      message.timestamp_sample = timestamp_.load();

      message.local_frame = 0;

      message.x = x;
      message.y = y;
      message.z = z;

      message.q[0] = NAN; // if you have quaternion values, include here. Else set 1st element to NAN
      message.q_offset[0] = NAN;

      message.vx = NAN;
      message.vy = NAN;
      message.vz = NAN;

      message.rollspeed = NAN;
      message.pitchspeed = NAN;
      message.yawspeed = NAN;


      // RCLCPP_INFO(this->get_logger(), "Publishing: '%f' '%f' '%f'", message.x, message.y, message.z);
      px4_publisher_->publish(message);
    }

    void topic_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
      
      // RCLCPP_INFO(this->get_logger(), "I heard: '%f'", msg->pose.position.x);
      x = msg->pose.position.x;
      y = msg->pose.position.y;
      z = msg->pose.position.z;

    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<px4_msgs::msg::VehicleVisualOdometry>::SharedPtr px4_publisher_;
    rclcpp::Subscription<px4_msgs::msg::Timesync>::SharedPtr timesync_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vrpn_subscription_;

    std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

    size_t count_;

    float x,y,z,vx,vy,vz,rollspeed,pitchspeed,yawspeed;

    Eigen::Vector4f q, q_offset; 
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MocapPublisher>());
  rclcpp::shutdown();
  return 0;
}