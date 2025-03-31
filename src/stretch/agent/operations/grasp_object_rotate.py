#!/usr/bin/env python3
"""
Revised GraspObjectOperation class with added base rotation logic.
"""

import os
import time
import timeit
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.core.interfaces import Observations
from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.filters import MaskTemporalFilter
from stretch.utils.geometry import point_global_to_base
from stretch.utils.gripper import GripperArucoDetector
from stretch.utils.point_cloud import show_point_cloud


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera, with added base rotation to align with target."""

    use_pitch_from_vertical: bool = True
    lift_distance: float = 0.2
    servo_to_grasp: bool = False
    _success: bool = False
    talk: bool = True
    verbose: bool = False

    offset_from_vertical = -np.pi / 2 - 0.1

    # Task information
    match_method: str = "class"
    target_object: Optional[str] = None
    _object_xyz: Optional[np.ndarray] = None

    # Should we use the previous mask at all?
    use_prev_mask: bool = False

    # Debugging UI elements
    show_object_to_grasp: bool = False
    show_servo_gui: bool = False
    show_point_cloud: bool = False
    debug_grasping: bool = False

    # This will delete the object from instance memory/voxel map after grasping
    delete_object_after_grasp: bool = False

    # Should we try grasping open-loop or not?
    _try_open_loop: bool = False

    # ------------------------
    # These are the most important parameters for tuning to make the grasping "feel" nice
    # Thresholds for centering on object
    align_x_threshold: int = 30
    align_y_threshold: int = 25

    # This is the distance before we start servoing to the object
    pregrasp_distance_from_object: float = 0.3

    # ------------------------
    # Grasping motion planning parameters and offsets
    median_distance_when_grasping: float = 0.18
    lift_min_height: float = 0.1
    lift_max_height: float = 1.0
    grasp_distance = 0.14

    # Movement parameters
    lift_arm_ratio: float = 0.05
    base_x_step: float = 0.10
    # New parameter: base rotation adjustment (radians per unit error)
    base_rotation_step: float = 0.05
    wrist_pitch_step: float = 0.2

    # Tracked object features for making sure we are grabbing the right thing
    tracked_object_features: Optional[torch.Tensor] = None

    # Parameters about how to grasp - less important
    grasp_loose: bool = False
    reset_observation: bool = False
    _grasp_arm_offset: float = 0.0
    _grasp_lift_offset: float = 0.0

    # Visual servoing config
    track_image_center: bool = False
    gripper_aruco_detector: GripperArucoDetector = None
    min_points_to_approach: int = 100
    detected_center_offset_x: int = 0
    detected_center_offset_y: int = 0
    percentage_of_image_when_grasping: float = 0.2
    open_loop_z_offset: float = -0.1
    open_loop_x_offset: float = -0.1
    max_failed_attempts: int = 10
    max_random_motions: int = 10

    # Timing issues
    expected_network_delay = 0.1
    open_loop: bool = False

    # Observation memory
    observations = MaskTemporalFilter(
        observation_history_window_size_secs=5.0, observation_history_window_size_n=3
    )

    def configure(
        self,
        target_object: Optional[str] = None,
        object_xyz: Optional[np.ndarray] = None,
        show_object_to_grasp: bool = False,
        servo_to_grasp: bool = True,
        show_servo_gui: bool = True,
        show_point_cloud: bool = False,
        reset_observation: bool = False,
        grasp_loose: bool = False,
        talk: bool = True,
        match_method: str = "class",
        delete_object_after_grasp: bool = True,
        try_open_loop: bool = False,
    ):
        """Configure the operation with the given keyword arguments."""
        if target_object is not None:
            self.target_object = target_object
        if object_xyz is not None:
            assert len(object_xyz) == 3, "Object xyz must be a 3D point."
            self._object_xyz = object_xyz
        self.show_object_to_grasp = show_object_to_grasp
        self.servo_to_grasp = servo_to_grasp
        self.show_servo_gui = show_servo_gui
        self.show_point_cloud = show_point_cloud
        self.reset_observation = reset_observation
        self.delete_object_after_grasp = delete_object_after_grasp
        self.grasp_loose = grasp_loose
        self.talk = talk
        self.match_method = match_method
        self._try_open_loop = try_open_loop
        if self.match_method not in ["class", "feature"]:
            raise ValueError(
                f"Unknown match method {self.match_method}. Should be 'class' or 'feature'."
            )

    def _debug_show_point_cloud(self, servo: Observations, current_xyz: np.ndarray) -> None:
        """Show the point cloud for debugging purposes."""
        world_xyz = servo.get_ee_xyz_in_world_frame()
        world_xyz_head = servo.get_xyz_in_world_frame()
        all_xyz = np.concatenate([world_xyz_head.reshape(-1, 3), world_xyz.reshape(-1, 3)], axis=0)
        all_rgb = np.concatenate([servo.rgb.reshape(-1, 3), servo.ee_rgb.reshape(-1, 3)], axis=0)
        show_point_cloud(all_xyz, all_rgb / 255, orig=current_xyz)

    def can_start(self):
        """Grasping can start if we have a target object and the robot is ready."""
        if self.target_object is None:
            self.error("No target object set.")
            return False

        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()

        return (
            self.agent.current_object is not None or self._object_xyz is not None
        ) and self.robot.in_manipulation_mode()

    def _compute_center_depth(
        self,
        servo: Observations,
        target_mask: np.ndarray,
        center_y: int,
        center_x: int,
        local_region_size: int = 5,
    ) -> float:
        """Compute the center depth of the object."""
        mask = np.zeros_like(target_mask)
        mask[
            max(center_y - local_region_size, 0) : min(center_y + local_region_size, mask.shape[0]),
            max(center_x - local_region_size, 0) : min(center_x + local_region_size, mask.shape[1]),
        ] = 1
        depth_mask = np.bitwise_and(servo.ee_depth > 1e-8, mask)
        depth = servo.ee_depth[target_mask & depth_mask]
        if len(depth) == 0:
            return 0.0
        median_depth = np.median(depth)
        return median_depth

    def get_class_mask(self, servo: Observations) -> np.ndarray:
        """Get the mask for the class of the object we are trying to grasp."""
        mask = np.zeros_like(servo.semantic).astype(bool)
        if self.verbose:
            print("[GRASP OBJECT] match method =", self.match_method)
        if self.match_method == "class":
            if self.agent.current_object is not None:
                target_class_id = self.agent.current_object.category_id
                target_class = self.agent.semantic_sensor.get_class_name_for_id(target_class_id)
            else:
                target_class = self.target_object
            if self.verbose:
                print("[GRASP OBJECT] Detecting objects of class", target_class)
            for iid in np.unique(servo.semantic):
                name = self.agent.semantic_sensor.get_class_name_for_id(iid)
                if name is not None and target_class in name:
                    mask = np.bitwise_or(mask, servo.semantic == iid)
        elif self.match_method == "feature":
            if self.target_object is None:
                raise ValueError(
                    f"Target object must be set before running match method {self.match_method}."
                )
            if self.verbose:
                print("[GRASP OBJECT] Detecting objects described as", self.target_object)
            text_features = self.agent.encode_text(self.target_object)
            best_score = float("-inf")
            best_iid = None
            all_matches = []
            for iid in np.unique(servo.instance):
                if iid < 0:
                    continue
                rgb = servo.ee_rgb * (servo.instance == iid)[:, :, None].repeat(3, axis=-1)
                features = self.agent.encode_image(rgb)
                score = self.agent.compare_features(text_features, features).item()
                print(f" - Score for {iid} is {score} / {self.agent.grasp_feature_match_threshold}.")
                if score > best_score:
                    best_score = score
                    best_iid = iid
                if score > self.agent.feature_match_threshold:
                    all_matches.append((score, iid, features))
            if len(all_matches) > 0:
                print("[MASK SELECTION] All matches:")
                for score, iid, features in all_matches:
                    print(f" - Matched {iid} with score {score}.")
            if len(all_matches) == 0:
                print("[MASK SELECTION] No matches found.")
            elif len(all_matches) == 1:
                print("[MASK SELECTION] One match found. We are done.")
                mask = servo.instance == best_iid
                self.tracked_object_features = all_matches[0][2]
            else:
                if self.tracked_object_features is not None:
                    best_score = float("-inf")
                    best_iid = None
                    best_features = None
                    for _, iid, features in all_matches:
                        score = self.agent.compare_features(self.tracked_object_features, features)
                        if score > best_score:
                            best_score = score
                            best_iid = iid
                            best_features = features
                    self.tracked_object_features = best_features
                else:
                    best_score = float("-inf")
                    best_iid = None
                    for score, iid, _ in all_matches:
                        if score > best_score:
                            best_score = score
                            best_iid = iid
                mask = servo.instance == best_iid
        else:
            raise ValueError(f"Invalid matching method {self.match_method}.")
        return mask

    def set_target_object_class(self, target_object: str):
        """Set the target object class."""
        self.target_object = target_object

    def reset(self):
        """Reset the operation."""
        self._success = False
        self.tracked_object_features = None
        self.observations.clear_history()

    def get_target_mask(
        self,
        servo: Observations,
        center: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Get target mask to move to."""
        class_mask = self.get_class_mask(servo)
        instance_mask = servo.instance
        if servo.ee_xyz is None:
            servo.compute_ee_xyz()
        target_mask = None
        target_mask_pts = float("-inf")
        maximum_overlap_mask = None
        maximum_overlap_pts = float("-inf")
        center_x, center_y = center
        for iid in np.unique(instance_mask):
            current_instance_mask = instance_mask == iid
            if class_mask[center_y, center_x] > 0 and current_instance_mask[center_y, center_x] > 0:
                print("!!! CENTERED ON THE RIGHT OBJECT !!!")
                return current_instance_mask
            mask = np.bitwise_and(current_instance_mask, class_mask)
            num_pts = sum(mask.flatten())
            if num_pts > target_mask_pts:
                target_mask = mask
                target_mask_pts = num_pts
        if maximum_overlap_pts > self.min_points_to_approach:
            return maximum_overlap_mask
        if target_mask is not None:
            return target_mask
        else:
            return None

    def sayable_target_object(self) -> str:
        """Return a sayable target object name."""
        return self.target_object.replace("_", " ")

    def _grasp(self, distance: Optional[float] = None) -> bool:
        """Helper function to close gripper around object."""
        self.cheer("Grasping object!")
        if self.talk:
            self.agent.robot_say(f"Grasping the {self.sayable_target_object()}!")
        print("Distance:", distance)
        joint_state = self.robot.get_joint_positions()
        if not self.open_loop or distance is not None:
            base_x = joint_state[HelloStretchIdx.BASE_X]
            wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
            arm = joint_state[HelloStretchIdx.ARM]
            lift = joint_state[HelloStretchIdx.LIFT]
            if distance is not None:
                distance = max(distance - self.grasp_distance, 0)
                print("Distance to move:", distance)
                if distance > 0:
                    arm_component = np.cos(wrist_pitch) * distance
                    lift_component = np.sin(wrist_pitch) * distance
                else:
                    arm_component = 0
                    lift_component = 0
            else:
                arm_component = 0
                lift_component = 0
            self.robot.arm_to(
                [
                    base_x,
                    np.clip(
                        lift + lift_component,
                        min(joint_state[HelloStretchIdx.LIFT], self.lift_min_height),
                        self.lift_max_height,
                    ),
                    arm + arm_component,
                    0,
                    wrist_pitch,
                    0,
                ],
                head=constants.look_at_ee,
                blocking=True,
            )
            time.sleep(0.1)
        self.robot.close_gripper(loose=self.grasp_loose, blocking=True)
        time.sleep(0.1)
        joint_state = self.robot.get_joint_positions()
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] += 0.2
        self.robot.arm_to(lifted_joint_state, head=constants.look_at_ee, blocking=True)
        return True

    def blue_highlight_mask(self, img):
        """Get a binary mask for the blue highlights in the image."""
        blue_condition = img[:, :, 2] > 100
        red_condition = img[:, :, 0] < 50
        green_condition = img[:, :, 1] < 50
        mask = blue_condition & red_condition & green_condition
        return mask.astype(np.uint8)

    def visual_servo_to_object(
        self, instance: Instance, max_duration: float = 120.0, max_not_moving_count: int = 50
    ) -> bool:
        """Use visual servoing to grasp the object, including rotating the base for alignment."""
        if instance is not None:
            self.intro(f"Visual servoing to grasp object {instance.global_id} {instance.category_id=}.")
        else:
            self.intro("Visual servoing to grasp {self.target_object} at {self._object_xyz}.")
        if self.show_servo_gui:
            self.warn("If you want to stop the visual servoing with the GUI up, press 'q'.")
        t0 = timeit.default_timer()
        aligned_once = False
        success = False
        prev_lift = float("Inf")
        if self.gripper_aruco_detector is None:
            self.gripper_aruco_detector = GripperArucoDetector()
        current_xyz = None
        failed_counter = 0
        not_moving_count = 0
        q_last = np.array([0.0 for _ in range(11)])
        random_motion_counter = 0
        center_depth = None
        prev_center_depth = None
        self.pregrasp_open_loop(
            self.get_object_xyz(), distance_from_object=self.pregrasp_distance_from_object
        )
        time.sleep(0.25)
        self.warn("Starting visual servoing.")
        if self.debug_grasping:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            debug_dir_name = f"debug/debug_{current_time}"
            os.mkdir(debug_dir_name)
        iter_ = 0
        while timeit.default_timer() - t0 < max_duration:
            servo = self.robot.get_servo_observation()
            joint_state = self.robot.get_joint_positions()
            world_xyz = servo.get_ee_xyz_in_world_frame()
            if not self.open_loop:
                base_x = joint_state[HelloStretchIdx.BASE_X]
                wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
                arm = joint_state[HelloStretchIdx.ARM]
                lift = joint_state[HelloStretchIdx.LIFT]
            if self.track_image_center:
                center_x, center_y = servo.ee_rgb.shape[1] // 2, servo.ee_rgb.shape[0] // 2
            else:
                center = self.gripper_aruco_detector.detect_center(servo.ee_rgb)
                if center is not None:
                    center_y, center_x = np.round(center).astype(int)
                    center_y += self.detected_center_offset_y
                else:
                    center_x, center_y = servo.ee_rgb.shape[1] // 2, servo.ee_rgb.shape[0] // 2
            center_x += self.detected_center_offset_x
            servo = self.agent.semantic_sensor.predict(servo, ee=True)
            latest_mask = self.get_target_mask(servo, center=(center_x, center_y))
            kernel = np.ones((3, 3), np.uint8)
            mask_np = latest_mask.astype(np.uint8)
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
            latest_mask = dilated_mask.astype(bool)
            self.observations.push_mask_to_observation_history(
                observation=latest_mask,
                timestamp=time.time(),
                mask_size_threshold=self.min_points_to_approach,
                acquire_lock=True,
            )
            target_mask = self.observations.get_latest_observation()
            if target_mask is None:
                target_mask = np.zeros([servo.ee_rgb.shape[0], servo.ee_rgb.shape[1]], dtype=bool)
            if center_depth is not None and center_depth > 1e-8:
                prev_center_depth = center_depth
            center_depth = self._compute_center_depth(servo, target_mask, center_y, center_x)
            if self.debug_grasping:
                mask = target_mask.astype(np.uint8) * 255
                debug_viz = np.zeros((240, 640, 3))
                debug_viz[:, :320, :] = servo.ee_rgb
                debug_viz[:, 320:, 0] = mask
                debug_viz[:, 320:, 1] = mask
                debug_viz[:, 320:, 2] = mask
                Image.fromarray(debug_viz.astype("uint8")).save(
                    f"{debug_dir_name}/img_{iter_:03d}.png"
                )
            iter_ += 1
            mask_center = self.observations.get_latest_centroid()
            if mask_center is None:
                failed_counter += 1
                if failed_counter < self.max_failed_attempts:
                    mask_center = np.array([center_y, center_x])
                else:
                    self.error(f"Lost track. Trying to grasp at {current_xyz}.")
                    if current_xyz is not None:
                        current_xyz[0] += self.open_loop_x_offset
                        current_xyz[2] += self.open_loop_z_offset
                    if self.show_servo_gui and not self.headless_machine:
                        cv2.destroyAllWindows()
                    if self._try_open_loop:
                        return self.grasp_open_loop(current_xyz)
                    else:
                        if self.talk:
                            self.agent.robot_say(f"I can't see the {self.target_object}.")
                        self._success = False
                        return False
                continue
            else:
                failed_counter = 0
                mask_center = mask_center.astype(int)
                assert (
                    world_xyz.shape[0] == servo.semantic.shape[0]
                    and world_xyz.shape[1] == servo.semantic.shape[1]
                ), "World xyz shape does not match semantic shape."
                current_xyz = world_xyz[int(mask_center[0]), int(mask_center[1])]
                if self.show_point_cloud:
                    self._debug_show_point_cloud(servo, current_xyz)
            if self.show_servo_gui and not self.headless_machine:
                print(" -> Displaying visual servoing GUI.")
                servo_ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_RGB2BGR)
                mask = target_mask.astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask[:, :, 0] = 0
                servo_ee_rgb = cv2.addWeighted(servo_ee_rgb, 0.5, mask, 0.5, 0, servo_ee_rgb)
                servo_ee_rgb = cv2.circle(servo_ee_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
                servo_ee_rgb = cv2.circle(
                    servo_ee_rgb, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )
                viz_ee_depth = cv2.normalize(servo.ee_depth, None, 0, 255, cv2.NORM_MINMAX)
                viz_ee_depth = viz_ee_depth.astype(np.uint8)
                viz_ee_depth = cv2.applyColorMap(viz_ee_depth, cv2.COLORMAP_JET)
                viz_ee_depth = cv2.circle(
                    viz_ee_depth, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )
                viz_image = np.concatenate([servo_ee_rgb, viz_ee_depth], axis=1)
                cv2.namedWindow("Visual Servoing", cv2.WINDOW_NORMAL)
                cv2.imshow("Visual Servoing", viz_image)
                cv2.waitKey(1)
                res = cv2.waitKey(1) & 0xFF
                if res == ord("q"):
                    break
            if self.debug_grasping:
                concatenated_image = np.concatenate((debug_viz.astype("uint8"), viz_image), axis=1)
                Image.fromarray(concatenated_image).save(
                    f"{debug_dir_name}/img_point_{iter_:03d}.png"
                )
            if not_moving_count > max_not_moving_count:
                self.info("Not moving; try to grasp.")
                success = self._grasp()
                break
            if target_mask is not None:
                object_depth = servo.ee_depth[target_mask]
                median_object_depth = np.median(servo.ee_depth[target_mask])
            else:
                if center_depth < self.median_distance_when_grasping and center_depth > 1e-8:
                    success = self._grasp(distance=center_depth)
                else:
                    failed_counter += 1
                continue
            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # Instead of adjusting the base translation for horizontal error, compute a rotation adjustment.
            if abs(dx) > self.align_x_threshold:
                rotation_adjustment = -self.base_rotation_step * (dx / self.align_x_threshold)
                print("Rotating base by:", rotation_adjustment, "radians")
                self.robot.move_base_to([0, 0, rotation_adjustment], relative=True)
            # Retain vertical correction via wrist pitch.
            if dy > self.align_y_threshold:
                wrist_pitch += -self.wrist_pitch_step * (dy / self.align_y_threshold)
            elif dy < -self.align_y_threshold:
                wrist_pitch += self.wrist_pitch_step * (abs(dy) / self.align_y_threshold)
            print("Adjusted parameters:")
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)
            self.robot.arm_to(
                [base_x, lift, arm, 0, wrist_pitch, 0],
                head=constants.look_at_ee,
                blocking=True,
            )
            prev_lift = lift
            time.sleep(self.expected_network_delay)
            q = [
                base_x,
                0.0,
                0.0,
                lift,
                arm,
                0.0,
                0.0,
                wrist_pitch,
                -0.5,
                0.0,
                0.0,
            ]
            q = np.array(q)
            ee_pos, ee_quat = self.robot_model.manip_fk(q)
            while ee_pos[2] < 0.03:
                lift += 0.01
                q[HelloStretchIdx.LIFT] = lift
                ee_pos, ee_quat = self.robot_model.manip_fk(q)
            if np.linalg.norm(q - q_last) < 0.05:
                not_moving_count += 1
            else:
                not_moving_count = 0
            q_last = q
            if random_motion_counter > self.max_random_motions:
                self.error("Failed to align to object after 10 random motions.")
                break
        if self.show_servo_gui and not self.headless_machine:
            cv2.destroyAllWindows()
        return success

    def get_object_xyz(self) -> np.ndarray:
        """Get the object xyz location."""
        if self._object_xyz is None:
            object_xyz = self.agent.current_object.get_median()
        else:
            object_xyz = self._object_xyz
        return object_xyz

    def run(self) -> None:
        self.intro("Grasping the object.")
        self._success = False
        if self.show_object_to_grasp:
            self.show_instance(self.agent.current_object)
        self.reset()
        assert self.target_object is not None, "Target object must be set before running."
        self.robot.open_gripper(blocking=True)
        obs = self.robot.get_observation()
        joint_state = self.robot.get_joint_positions()
        model = self.robot.get_robot_model()
        if joint_state[HelloStretchIdx.GRIPPER] < 0.0:
            self.robot.open_gripper(blocking=True)
        xyt = self.robot.get_base_pose()
        object_xyz = self.get_object_xyz()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)
        if self.use_pitch_from_vertical:
            obs = self.robot.get_observation()
            joint_state = obs.joint
            model = self.robot.get_robot_model()
            ee_pos, ee_rot = model.manip_fk(joint_state)
            pose = np.eye(4)
            pose[:3, :3] = R.from_quat(ee_rot).as_matrix()
            pose[:3, 3] = ee_pos
            delta = np.eye(4)
            delta[2, 3] = -0.3
            pose = np.dot(pose, delta)
            ee_pos = pose[:3, 3]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
        else:
            pitch_from_vertical = 0.0
        joint_state[HelloStretchIdx.WRIST_PITCH] = self.offset_from_vertical + pitch_from_vertical
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)
        if self.servo_to_grasp:
            self._success = self.visual_servo_to_object(self.agent.current_object)
        if self.reset_observation:
            self.agent.reset_object_plans()
            self.agent.get_voxel_map().instances.pop_global_instance(
                env_id=0, global_instance_id=self.agent.current_object.global_id
            )
        if self.delete_object_after_grasp:
            voxel_map = self.agent.get_voxel_map()
            if voxel_map is not None:
                voxel_map.delete_instance(self.agent.current_object, assume_explored=False)
        if self.talk and self._success:
            self.agent.robot_say(f"I think I grasped the {self.sayable_target_object()}.")
        self.robot.move_to_manip_posture()

    def pregrasp_open_loop(self, object_xyz: np.ndarray, distance_from_object: float = 0.35):
        """Move to a pregrasp position in an open loop manner."""
        xyt = self.robot.get_base_pose()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)
        joint_state = self.robot.get_joint_positions()
        model = self.robot.get_robot_model()
        ee_pos, ee_rot = model.manip_fk(joint_state)
        rotation = R.from_quat(ee_rot)
        rotation = rotation.as_euler("xyz")
        print("Rotation", rotation)
        if rotation[1] > np.pi / 4:
            rotation[1] = np.pi / 4
        old_ee_rot = ee_rot
        ee_rot = R.from_euler("xyz", rotation).as_quat()
        vector_to_object = relative_object_xyz - ee_pos
        vector_to_object = vector_to_object / np.linalg.norm(vector_to_object)
        vector_to_object[2] = max(vector_to_object[2], vector_to_object[1])
        print("Absolute object xyz was:", object_xyz)
        print("Relative object xyz was:", relative_object_xyz)
        shifted_object_xyz = relative_object_xyz - (distance_from_object * vector_to_object)
        print("Pregrasp xyz:", shifted_object_xyz)
        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            shifted_object_xyz, ee_rot, q0=joint_state
        )
        print("Pregrasp joint positions: ")
        print(" - arm: ", target_joint_positions[HelloStretchIdx.ARM])
        print(" - lift: ", target_joint_positions[HelloStretchIdx.LIFT])
        print(" - roll: ", target_joint_positions[HelloStretchIdx.WRIST_ROLL])
        print(" - pitch: ", target_joint_positions[HelloStretchIdx.WRIST_PITCH])
        print(" - yaw: ", target_joint_positions[HelloStretchIdx.WRIST_YAW])
        if not success:
            print("Failed to find a valid IK solution.")
            self._success = False
            return
        elif (
            target_joint_positions[HelloStretchIdx.ARM] < -0.05
            or target_joint_positions[HelloStretchIdx.LIFT] < -0.05
        ):
            print(
                f"{self.name}: Target joint state is invalid: {target_joint_positions}. Positions for arm and lift must be positive."
            )
            self._success = False
            return
        target_joint_positions[HelloStretchIdx.ARM] = max(target_joint_positions[HelloStretchIdx.ARM], 0)
        target_joint_positions[HelloStretchIdx.LIFT] = max(target_joint_positions[HelloStretchIdx.LIFT], 0)
        target_joint_positions[HelloStretchIdx.WRIST_YAW] = 0
        target_joint_positions[HelloStretchIdx.WRIST_ROLL] = 0
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance
        print(f"{self.name}: Moving to pre-grasp position.")
        self.robot.arm_to(target_joint_positions, head=constants.look_at_ee, blocking=True)
        print("... done.")

    def grasp_open_loop(self, object_xyz: np.ndarray):
        """Grasp the object in an open loop manner."""
        model = self.robot.get_robot_model()
        xyt = self.robot.get_base_pose()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)
        joint_state = self.robot.get_joint_positions()
        ee_pos, ee_rot = model.manip_fk(joint_state)
        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            relative_object_xyz, ee_rot, q0=joint_state
        )
        target_joint_positions[HelloStretchIdx.BASE_X] -= 0.04
        if not success:
            print("Failed to find a valid IK solution.")
            self._success = False
            return
        elif (
            target_joint_positions[HelloStretchIdx.ARM] < 0
            or target_joint_positions[HelloStretchIdx.LIFT] < 0
        ):
            print(
                f"{self.name}: Target joint state is invalid: {target_joint_positions}. Positions for arm and lift must be positive."
            )
            self._success = False
            return
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance
        print(f"{self.name}: Moving to grasp position.")
        self.robot.arm_to(target_joint_positions, head=constants.look_at_ee, blocking=True)
        time.sleep(0.5)
        print(f"{self.name}: Closing the gripper.")
        self.robot.close_gripper(blocking=True)
        time.sleep(0.5)
        print(f"{self.name}: Lifting the arm up so as not to hit the base.")
        self.robot.arm_to(target_joint_positions_lifted, head=constants.look_at_ee, blocking=True)
        print(f"{self.name}: Return arm to initial configuration.")
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)
        print(f"{self.name}: Done.")
        self._success = True
        return

    def was_successful(self) -> bool:
        """Return true if successful."""
        return self._success
