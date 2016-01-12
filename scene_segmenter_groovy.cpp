#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/shared_ptr.hpp>
#include <pcl_ros/point_cloud.h>

#include <pcl/visualization/cloud_viewer.h>

#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "perception_msgs/SegmentedObject.h"
#include "perception_msgs/SegmentedObjectList.h"
#include "perception_msgs/ObjectCenterProperty.h"

#include "cluster_extractor.h"



namespace scene_segmenter_node
{
    class SceneSegmenterNode
    {
        private:
            ros::NodeHandle node_handle;

            ros::Subscriber pointCloudSubscriber;
            void pointCloudMessageCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

            ros::Publisher segmentedObjectsPublisher;

        public:
            SceneSegmenterNode();
    };


    SceneSegmenterNode::SceneSegmenterNode(): node_handle("")
    {
        pointCloudSubscriber = node_handle.subscribe("/camera/depth/points/", 10, &SceneSegmenterNode::pointCloudMessageCallback, this);

        segmentedObjectsPublisher = node_handle.advertise<perception_msgs::SegmentedObjectList>("segmented_objects",10);

        ROS_INFO("scene_segmenter_node ready");
    }


    void SceneSegmenterNode::pointCloudMessageCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr sceneCloud (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *sceneCloud);

        //extract clusters
        ClusterExtractor *clusterExtractor = new ClusterExtractor();
        clusterExtractor->setCloud(sceneCloud);
        clusterExtractor->computeClusters();
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = clusterExtractor->getCloudClusters();
        delete clusterExtractor;

        //publish clusters
        perception_msgs::SegmentedObjectList segmentedObjectsList;
        for(int i = 0; i<cloudClusters.size();i++)
        {
            //convert segmented object point cloud to sensor_msgs point cloud
            sensor_msgs::PointCloud2 sensorMessagePointCloud;
            pcl::toROSMsg(*cloudClusters.at(i),sensorMessagePointCloud);

            perception_msgs::SegmentedObject segmentedObject;
            segmentedObject.segmentedObjectID = i;
            segmentedObject.segmentedObjectPointCloud = sensorMessagePointCloud;
            segmentedObject.segmentedObjectPointCloud.header.frame_id = msg.get()->header.frame_id;
            segmentedObjectsList.segmentedObjects.push_back(segmentedObject);
        }

        segmentedObjectsPublisher.publish(segmentedObjectsList);
    }
}



int main(int argc, char **argv) 
{

  ros::init(argc, argv, "scene_segmenter_node");
  ros::NodeHandle nh;

  scene_segmenter_node::SceneSegmenterNode node;

  ros::spin();
  return 0;
}