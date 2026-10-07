// Local RGB-D + IMU adapter for the Mechazo11 Jazzy port.
// Sensor timestamps are preserved. No GT data enters this node.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <array>

using Image = sensor_msgs::msg::Image;
using Imu = sensor_msgs::msg::Imu;
static double stamp(const builtin_interfaces::msg::Time &t) {return t.sec + t.nanosec*1e-9;}
static void pose_line(std::ostream& f, double t, const Sophus::SE3f& Twc) {
    const auto p=Twc.translation(); const auto q=Twc.unit_quaternion();
    f << std::fixed << std::setprecision(9) << t << " " << p.x() << " " << p.y() << " " << p.z()
      << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}
class OfficeRgbd : public rclcpp::Node {
public:
    OfficeRgbd(): Node("office_orbslam3") {
        auto voc=declare_parameter<std::string>("vocabulary", "");
        auto config=declare_parameter<std::string>("settings", "");
        out_=declare_parameter<std::string>("output_dir", "");
        inertial_=declare_parameter<bool>("use_imu", true);
        auto viewer=declare_parameter<bool>("viewer", false);
        tolerance_=declare_parameter<double>("sync_tolerance", .015);
        if(voc.empty()||config.empty()||out_.empty())throw std::runtime_error("vocabulary, settings and output_dir required");
        std::filesystem::create_directories(out_);
        csv_.open(out_+"/tracking.csv");
        csv_ << "timestamp,depth_timestamp,state,imu_initialized,map_id,imu_batch,tracked_points,tracking_ms\n";
        online_.open(out_+"/online_camera.tum");online_ << "# timestamp tx ty tz qx qy qz qw; camera optical frame; see tracking.csv for map resets\n";
        cv::setNumThreads(2);
        slam_=std::make_unique<ORB_SLAM3::System>(voc,config,inertial_?ORB_SLAM3::System::IMU_RGBD:ORB_SLAM3::System::RGBD,viewer);
        auto qos=rclcpp::QoS(100).reliable();
        rgb_sub_=create_subscription<Image>("/d400/color/image_raw",qos,[this](Image::ConstSharedPtr m){std::lock_guard<std::mutex> g(mu_);rgb_.push_back(m);++rgb_count_;cv_.notify_one();});
        depth_sub_=create_subscription<Image>("/d400/aligned_depth_to_color/image_raw",qos,[this](Image::ConstSharedPtr m){std::lock_guard<std::mutex> g(mu_);depth_.push_back(m);++depth_count_;cv_.notify_one();});
        if(inertial_)imu_sub_=create_subscription<Imu>("/imu",rclcpp::QoS(4000).reliable(),[this](Imu::ConstSharedPtr m){std::lock_guard<std::mutex> g(mu_);imu_.push_back(m);++imu_count_;cv_.notify_one();});
        pub_=create_publisher<nav_msgs::msg::Odometry>("/orb_slam3/odometry",10);
        status(false);
        worker_=std::thread(&OfficeRgbd::process,this);
        RCLCPP_INFO(get_logger(),"READY: mode=%s, RGB-D timestamp tolerance=%.3f s",inertial_?"RGB-D-Inertial":"RGB-D",tolerance_);
    }
    ~OfficeRgbd(){finish();}
    void finish() {
        if(finished_.exchange(true))return;
        stop_=true;cv_.notify_all();if(worker_.joinable())worker_.join();
        if(!slam_)return;
        slam_->Shutdown();
        auto atlas=slam_->GetAtlasForDiagnostics();auto maps=atlas->GetAllMaps();
        std::ofstream manifest(out_+"/maps.csv");manifest<<"map_id,keyframes,points,imu_initialized,inertial_BA1,inertial_BA2,selected\n";
        ORB_SLAM3::Map* largest=nullptr;size_t max_kf=0;
        for(auto map:maps) {auto n=map->GetAllKeyFrames().size();if(n>max_kf){max_kf=n;largest=map;}}
        for(auto map:maps) {
            auto kfs=map->GetAllKeyFrames();auto pts=map->GetAllMapPoints();
            manifest<<map->GetId()<<","<<kfs.size()<<","<<pts.size()<<","<<map->isImuInitialized()<<","<<map->GetIniertialBA1()<<","<<map->GetIniertialBA2()<<","<<(map==largest)<<"\n";
            if(kfs.empty())continue;
            std::string prefix=out_+"/map_"+std::to_string(map->GetId());
            slam_->SaveCameraTrajectoryForMap(prefix+"_camera.tum",map);
            std::sort(kfs.begin(),kfs.end(),[](auto a,auto b){return a->mTimeStamp<b->mTimeStamp;});
            std::ofstream kfout(prefix+"_keyframes_camera.tum");
            for(auto kf:kfs)if(!kf->isBad())pose_line(kfout,kf->mTimeStamp,kf->GetPoseInverse());
            kfout.close();
            if(map==largest) {
                std::filesystem::copy_file(prefix+"_camera.tum",out_+"/trajectory_camera.tum",std::filesystem::copy_options::overwrite_existing);
                std::filesystem::copy_file(prefix+"_keyframes_camera.tum",out_+"/keyframes_camera.tum",std::filesystem::copy_options::overwrite_existing);
                std::vector<Eigen::Vector3f> xyz;
                for(auto p:pts)if(p&&!p->isBad()) {auto v=p->GetWorldPos();if(v.allFinite())xyz.push_back(v);}
                std::ofstream ply(out_+"/sparse_map.ply");
                ply<<"ply\nformat ascii 1.0\nelement vertex "<<xyz.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
                for(const auto& v:xyz)ply<<v.x()<<" "<<v.y()<<" "<<v.z()<<"\n";
            }
        }
        manifest.close();csv_.flush();online_.flush();status(true);
        RCLCPP_INFO(get_logger(),"FINISHED: %zu frames processed, %zu OK, %zu initialized-IMU frames, %zu maps",processed_,ok_,imu_initialized_frames_,maps.size());
    }
private:
    void status(bool final) {
        size_t rgb,depth,imu,rq,dq,iq;{std::lock_guard<std::mutex> g(mu_);rgb=rgb_count_;depth=depth_count_;imu=imu_count_;rq=rgb_.size();dq=depth_.size();iq=imu_.size();}
        std::ofstream f(out_+"/progress.json.tmp");
        f<<"{\"final\":"<<(final?"true":"false")<<",\"mode\":\""<<(inertial_?"RGBD_IMU":"RGBD")<<"\",\"rgb_received\":"<<rgb<<",\"depth_received\":"<<depth<<",\"imu_received\":"<<imu
         <<",\"frames_processed\":"<<processed_<<",\"tracking_ok\":"<<ok_<<",\"imu_initialized_frames\":"<<imu_initialized_frames_
         <<",\"imu_measurements_submitted\":"<<imu_submitted_<<",\"unpaired_images\":"<<unpaired_<<",\"rgb_queue\":"<<rq<<",\"depth_queue\":"<<dq<<",\"imu_queue\":"<<iq<<",\"state_counts\":[";
        for(size_t i=0;i<states_.size();++i)f<<(i?",":"")<<states_[i];
        f<<"]}\n";f.close();std::filesystem::rename(out_+"/progress.json.tmp",out_+"/progress.json");
    }
    void process() {
      try {
        while(!stop_) {
            Image::ConstSharedPtr rgb,dep;std::vector<ORB_SLAM3::IMU::Point> batch;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_.wait(g,[this]{return stop_||(!rgb_.empty()&&!depth_.empty()&&(!inertial_||(!imu_.empty()&&stamp(imu_.back()->header.stamp)>=stamp(rgb_.front()->header.stamp))));});
                if(stop_)break;
                double tr=stamp(rgb_.front()->header.stamp),td=stamp(depth_.front()->header.stamp);
                if(tr-td>tolerance_){depth_.pop_front();++unpaired_;continue;}
                if(td-tr>tolerance_){rgb_.pop_front();++unpaired_;continue;}
                rgb=rgb_.front();dep=depth_.front();rgb_.pop_front();depth_.pop_front();
                while(inertial_&&!imu_.empty()&&stamp(imu_.front()->header.stamp)<=tr) {
                    auto m=imu_.front();imu_.pop_front();
                    batch.emplace_back(m->linear_acceleration.x,m->linear_acceleration.y,m->linear_acceleration.z,m->angular_velocity.x,m->angular_velocity.y,m->angular_velocity.z,stamp(m->header.stamp));
                }
            }
            const double t=stamp(rgb->header.stamp);
            auto im=cv_bridge::toCvShare(rgb,"rgb8");auto depth=cv_bridge::toCvShare(dep,"16UC1");
            auto begin=std::chrono::steady_clock::now();
            Sophus::SE3f Tcw=slam_->TrackRGBD(im->image,depth->image,t,batch);
            auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            int state=slam_->GetTrackingState();auto atlas=slam_->GetAtlasForDiagnostics();bool initialized=inertial_&&atlas->isImuInitialized();
            ++processed_;imu_submitted_+=batch.size();if(state>=0&&state<int(states_.size()))++states_[state];if(initialized)++imu_initialized_frames_;
            size_t tracked=0;for(auto p:slam_->GetTrackedMapPoints())if(p&&!p->isBad())++tracked;
            csv_<<std::fixed<<std::setprecision(9)<<t<<","<<stamp(dep->header.stamp)<<","<<state<<","<<initialized<<","<<atlas->GetCurrentMap()->GetId()<<","<<batch.size()<<","<<tracked<<","<<elapsed<<"\n";
            if(state==2) {
                ++ok_;auto Twc=Tcw.inverse();pose_line(online_,t,Twc);
                nav_msgs::msg::Odometry msg;msg.header=rgb->header;msg.header.frame_id="orb_map";msg.child_frame_id="d400_color";
                auto p=Twc.translation();auto q=Twc.unit_quaternion();msg.pose.pose.position.x=p.x();msg.pose.pose.position.y=p.y();msg.pose.pose.position.z=p.z();msg.pose.pose.orientation.x=q.x();msg.pose.pose.orientation.y=q.y();msg.pose.pose.orientation.z=q.z();msg.pose.pose.orientation.w=q.w();
                if(rclcpp::ok())pub_->publish(msg);
            }
            csv_.flush();online_.flush();status(false);
            if(processed_%30==0)RCLCPP_INFO(get_logger(),"Frames=%zu state=%d imu_initialized=%d map=%lu",processed_,state,initialized,atlas->GetCurrentMap()->GetId());
        }
      }catch(const std::exception& e){RCLCPP_ERROR(get_logger(),"Processing exception: %s",e.what());std::ofstream(out_+"/processing_error.txt")<<e.what();stop_=true;status(false);}
    }
    std::unique_ptr<ORB_SLAM3::System> slam_;std::string out_;bool inertial_;double tolerance_;
    std::mutex mu_;std::condition_variable cv_;std::thread worker_;std::atomic<bool> stop_{false},finished_{false};
    std::deque<Image::ConstSharedPtr> rgb_,depth_;std::deque<Imu::ConstSharedPtr> imu_;
    size_t rgb_count_=0,depth_count_=0,imu_count_=0,processed_=0,ok_=0,imu_initialized_frames_=0,imu_submitted_=0,unpaired_=0;
    std::array<size_t,8> states_{};std::ofstream csv_,online_;
    rclcpp::Subscription<Image>::SharedPtr rgb_sub_,depth_sub_;rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
};
int main(int argc,char**argv) {
    rclcpp::init(argc,argv);
    try {auto n=std::make_shared<OfficeRgbd>();rclcpp::spin(n);n->finish();n.reset();}
    catch(const std::exception& e){std::cerr<<"Fatal: "<<e.what()<<std::endl;if(rclcpp::ok())rclcpp::shutdown();return 1;}
    if(rclcpp::ok())rclcpp::shutdown();return 0;
}