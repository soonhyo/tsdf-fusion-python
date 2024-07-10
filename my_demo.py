#!/usr/bin/env python

import rospy
import time
import numpy as np
import cv2
import fusion
import tf
import struct
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

class TSDFReconstruction:
    def __init__(self):
        rospy.init_node('tsdf_reconstruction', anonymous=True)

        self.bridge = CvBridge()
        self.tsdf_vol = None
        self.cam_intr = None
        self.vol_bnds = np.zeros((3, 2))
        self.n_imgs = 0

        # Parameters
        self.max_depth = 0.5  # Maximum depth in meters

        # ROS publishers and subscribers
        self.pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_rect_color', Image, self.color_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()

        self.color_image = None
        self.depth_image = None

        self.process_frames()

    def camera_info_callback(self, msg):
        if self.cam_intr is None:
            self.cam_intr = np.array(msg.K).reshape(3, 3)

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_image = self.depth_image.astype(float) / 1000.0  # Convert to meters
        self.depth_image[self.depth_image > self.max_depth] = 0  # Apply max depth constraint

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def get_camera_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/world', '/camera', rospy.Time(0))
            trans_matrix = tf.transformations.translation_matrix(trans)
            rot_matrix = tf.transformations.quaternion_matrix(rot)
            cam_pose = np.dot(trans_matrix, rot_matrix)
            return cam_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    def process_frames(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            cam_pose = self.get_camera_pose()
            if self.color_image is not None and self.depth_image is not None and cam_pose is not None and self.cam_intr is not None:
                if self.tsdf_vol is None:
                    self.initialize_tsdf()

                self.integrate_frame(cam_pose)
                self.publish_pointcloud()
                self.reset_data()

            rate.sleep()

    def initialize_tsdf(self):
        self.estimate_volume_bounds()
        self.tsdf_vol = fusion.TSDFVolume(self.vol_bnds, voxel_size=0.005)

    def estimate_volume_bounds(self):
        rospy.loginfo("Estimating voxel volume bounds...")
        for _ in range(30):  # Estimate bounds from initial frames
            cam_pose = self.get_camera_pose()
            if self.depth_image is not None and cam_pose is not None:
                view_frust_pts = fusion.get_view_frustum(self.depth_image, self.cam_intr, cam_pose)
                self.vol_bnds[:, 0] = np.minimum(self.vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                self.vol_bnds[:, 1] = np.maximum(self.vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    def integrate_frame(self, cam_pose):
        rospy.loginfo("Integrating frame %d" % (self.n_imgs + 1))
        self.tsdf_vol.integrate(self.color_image, self.depth_image, self.cam_intr, cam_pose, obs_weight=5.0)
        self.n_imgs += 1

    def publish_pointcloud(self):
        rospy.loginfo("Publishing point cloud...")
        point_cloud = self.tsdf_vol.get_point_cloud()

        # Convert point cloud to PointCloud2 message with color
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        points = []
        for point in point_cloud:
            x, y, z, b, g, r = point
            rgb = struct.unpack('I', struct.pack('BBBB', int(b), int(g), int(r), 255))[0]
            points.append([x, y, z, rgb])

        pc2_msg = pc2.create_cloud(header, [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
        ], points)

        self.pointcloud_pub.publish(pc2_msg)

    def reset_data(self):
        self.color_image = None
        self.depth_image = None

if __name__ == "__main__":
    try:
        TSDFReconstruction()
    except rospy.ROSInterruptException:
        pass
