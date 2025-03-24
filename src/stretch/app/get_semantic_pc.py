#!/usr/bin/env python3

import time
import click
import cv2
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.point_cloud import show_point_cloud
from stretch.utils.image import adjust_gamma

def save_point_cloud(filename: str, points: np.ndarray, colors: np.ndarray):
    """
    Save point cloud data to a PLY file.
    
    Args:
        filename: Name of the file to save the point cloud.
        points: (N x 3) numpy array of XYZ coordinates.
        colors: (N x 3) numpy array of RGB colors (uint8).
    """
    num_points = points.shape[0]
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % num_points)
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write("%f %f %f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))

@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option("--local", is_flag=True, help="Set if executing on the robot rather than remotely")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--target_object", type=str, default="shoe", help="Target object to segment")
@click.option("--gamma", type=float, default=1.0, help="Gamma correction factor for head camera RGB images")
@click.option("--save_file", type=str, default="point_cloud.ply", help="Filename to save the point cloud")
def main(robot_ip: str, reset: bool, local: bool, parameter_file: str, target_object: str, gamma: float, save_file: str):
    """
    This script sets up the robot, captures head camera images, and:
      - Optionally resets the robot and moves it to a consistent posture.
      - Performs semantic segmentation on the head camera image.
      - Highlights the target object (specified by --target_object) in green.
      - Generates a point cloud from the head camera’s depth and RGB data (with segmentation applied).
      - Displays and saves the point cloud.
    """
    # Load configuration parameters and initialize the robot client.
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    
    # Reset and posture commands for a consistent view.
    if reset:
        demo = RobotAgent(robot, parameters, None)
        demo.start(visualize_map_at_start=False, can_move=True)
        demo.move_closed_loop([0, 0, 0], max_time=60.0)
        robot.move_to_manip_posture()
    else:
        print("Starting the robot...")
        robot.start()
        robot.move_to_manip_posture()
        robot.open_gripper()
        time.sleep(2)
        # Adjust arm position as needed to get a consistent view.
        robot.arm_to([0.0, 0.4, 0.05, 0, -np.pi / 4, 0], blocking=True)

    # Create the semantic sensor for segmentation (using the head camera).
    semantic_sensor = create_semantic_sensor(parameters, verbose=False)

    # For demonstration, define a mapping from target object names to segmentation label IDs.
    # Adjust these IDs based on your segmentation model.
    object_label_mapping = {
        "shoe": 3,
        "bottle": 1,
        "cup": 2,
        # add additional mappings as needed.
    }
    target_label = object_label_mapping.get(target_object, None)
    if target_label is None:
        print(f"Warning: No mapping found for target_object '{target_object}'. Target highlighting will be skipped.")

    # Wait for a valid servo observation containing head camera data.
    servo = None
    while servo is None:
        servo = robot.get_servo_observation()
        if servo is None:
            time.sleep(0.01)
    
    # Grab the head camera RGB image and apply gamma correction.
    head_rgb = servo.rgb.copy()
    if gamma != 1.0:
        head_rgb = adjust_gamma(head_rgb, gamma)
    
    # Run segmentation on the head camera image.
    _obs = semantic_sensor.predict(servo, ee=False)  # Use head camera (ee=False)
    if semantic_sensor.is_semantic():
        segmentation = _obs.semantic
    elif semantic_sensor.is_instance():
        segmentation = _obs.instance
    else:
        raise ValueError("Unknown segmentation model type")

    # Create an overlay: highlight target object's pixels in green.
    segmented_rgb = head_rgb.copy()
    if target_label is not None:
        mask = (segmentation == target_label)
        segmented_rgb[mask] = [0, 255, 0]

    # Generate the point cloud using head camera data.
    head_xyz = servo.get_xyz_in_world_frame().reshape(-1, 3)
    segmented_rgb_flat = segmented_rgb.reshape(-1, 3)

    # Display the point cloud.
    show_point_cloud(head_xyz, segmented_rgb_flat, orig=np.zeros(3))

    # Save the point cloud to a PLY file.
    print(f"Saving point cloud to {save_file} ...")
    save_point_cloud(save_file, head_xyz, segmented_rgb_flat)
    
    robot.stop()

if __name__ == "__main__":
    main()
