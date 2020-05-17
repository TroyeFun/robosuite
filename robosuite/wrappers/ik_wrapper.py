"""
This file implements a wrapper for controlling the robot through end effector
movements instead of joint velocities. This is useful in learning pipelines
that want to output actions in end effector space instead of joint space.
"""

import os
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper


class IKWrapper(Wrapper):
    env = None

    def __init__(self, env, action_repeat=1, use_abs_pose=False):
        """
        Initializes the inverse kinematics wrapper.
        This wrapper allows for controlling the robot through end effector
        movements instead of joint velocities.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            action_repeat (int): Determines the number of times low-level joint
                control actions will be commanded per high-level end effector
                action. Higher values will allow for more precise control of
                the end effector to the commanded targets.
            use_abs_pose (bool): True if take absolute pos and rotation w.r.t. base
                frame, else take relative pos and rotation w.r.t. current pose.
        """
        super().__init__(env)
        if self.env.mujoco_robot.name == "sawyer":
            from robosuite.controllers import SawyerIKController

            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
                use_abs_pos=use_abs_pose,
            )
        elif self.env.mujoco_robot.name == "baxter":
            from robosuite.controllers import BaxterIKController

            self.controller = BaxterIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        self.action_repeat = action_repeat
        self.use_abs_pose = use_abs_pose

    def set_robot_joint_positions(self, positions):
        """
        Overrides the function to set the joint positions directly, since we need to notify
        the IK controller of the change.
        """
        self.env.set_robot_joint_positions(positions)
        self.controller.sync_state()

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def reset(self):
        ret = super().reset()
        self.controller.sync_state()
        return ret

    def step(self, action):
        """
        Move the end effector(s) according to the input control.

        Args:
            action (numpy array): The array should have the corresponding elements.
                0-2: The desired change in end effector position in x, y, and z.
                3-6: The desired change in orientation, expressed as a (x, y, z, w) quaternion.
                    Note that this quaternion encodes a relative rotation with respect to the
                    current gripper orientation. If the current rotation is r, this corresponds
                    to a quaternion d such that r * d will be the new rotation.
                *: Controls for gripper actuation.

                Note: When wrapping around a Baxter environment, the indices 0-6 inidicate the
                right hand. Indices 7-13 indicate the left hand, and the rest (*) are the gripper
                inputs (first right, then left).
        """

        input_1 = self._make_input(action[:7], self.env._right_hand_quat)
        if self.env.mujoco_robot.name == "sawyer":
            velocities = self.controller.get_control(**input_1)
            low_action = np.concatenate([velocities, action[7:]])
        elif self.env.mujoco_robot.name == "baxter":
            input_2 = self._make_input(action[7:14], self.env._left_hand_quat)
            velocities = self.controller.get_control(input_1, input_2)
            low_action = np.concatenate([velocities, action[14:]])
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        # keep trying to reach the target in a closed-loop
        for i in range(self.action_repeat):
            ret = self.env.step(low_action)
            if i + 1 < self.action_repeat:
                velocities = self.controller.get_control()
                if self.env.mujoco_robot.name == "sawyer":
                    low_action = np.concatenate([velocities, action[7:]])
                elif self.env.mujoco_robot.name == "baxter":
                    low_action = np.concatenate([velocities, action[14:]])
                else:
                    raise Exception(
                        "Only Sawyer and Baxter robot environments are supported for IK "
                        "control currently."
                    )

        return ret

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        if self.use_abs_pose:
            return {
                "dpos": action[:3],
                # IK controller takes an absolute orientation in robot base frame
                "rotation": T.quat2mat(action[3:7]),
            }
        else:
            return {
                "dpos": action[:3],
                # IK controller takes an absolute orientation in robot base frame
                "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
            }

class DoubleModeIKWrapper(Wrapper):
    env = None

    def __init__(self, env, action_repeat=1, use_abs_pose=False, clip_vel=0):
        """
        Double mode version of IKWrapper, that is: if use_ik_mode (bool) is not specified in
        action (dict), then self.step(action) return self.env.step(action) directly.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            action_repeat (int): Determines the number of times low-level joint
                control actions will be commanded per high-level end effector
                action. Higher values will allow for more precise control of
                the end effector to the commanded targets.
            use_abs_pose (bool): True if take absolute pos and rotation w.r.t. base
                frame, else take relative pos and rotation w.r.t. current pose.
            clip_vel (positive float): clip velocities so that object in hand would not be thrown
        """
        super().__init__(env)
        if self.env.mujoco_robot.name == "sawyer":
            from robosuite.controllers import SawyerIKController

            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
                use_abs_pos=use_abs_pose,
            )
        elif self.env.mujoco_robot.name == "baxter":
            from robosuite.controllers import BaxterIKController

            self.controller = BaxterIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        self.action_repeat = action_repeat
        self.use_abs_pose = use_abs_pose
        self.clip_vel = clip_vel

    def set_robot_joint_positions(self, positions):
        """
        Overrides the function to set the joint positions directly, since we need to notify
        the IK controller of the change.
        """
        self.env.set_robot_joint_positions(positions)
        self.controller.sync_state()

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def reset(self):
        ret = super().reset()
        self.controller.sync_state()
        return ret

    def step(self, action):
        """
        Move the end effector(s) according to the input control.

        Args:
            action (numpy array): The array should have the corresponding elements.
                0-2: The desired change in end effector position in x, y, and z.
                3-6: The desired change in orientation, expressed as a (x, y, z, w) quaternion.
                    Note that this quaternion encodes a relative rotation with respect to the
                    current gripper orientation. If the current rotation is r, this corresponds
                    to a quaternion d such that r * d will be the new rotation.
                *: Controls for gripper actuation.

                Note: When wrapping around a Baxter environment, the indices 0-6 inidicate the
                right hand. Indices 7-13 indicate the left hand, and the rest (*) are the gripper
                inputs (first right, then left).
        """
        if not isinstance(action, dict):
            for i in range(self.action_repeat):
                ret = self.env.step(action)
            return ret
        else:
            action_dict = action
            action = action_dict['action']
            use_ik_mode = action_dict['use_ik_mode']
            if not use_ik_mode:
                for i in range(self.action_repeat):
                    ret = self.env.step(action)
                return ret

        input_1 = self._make_input(action[:7], self.env._right_hand_quat)
        if self.env.mujoco_robot.name == "sawyer":
            velocities = self.controller.get_control(**input_1)
            velocities = self._clip_velocity(velocities)
            low_action = np.concatenate([velocities, action[7:]])
        elif self.env.mujoco_robot.name == "baxter":
            input_2 = self._make_input(action[7:14], self.env._left_hand_quat)
            velocities = self.controller.get_control(input_1, input_2)
            velocities = self._clip_velocity(velocities)
            low_action = np.concatenate([velocities, action[14:]])
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        # keep trying to reach the target in a closed-loop
        for i in range(self.action_repeat):
            ret = self.env.step(low_action)
            if i + 1 < self.action_repeat:
                velocities = self.controller.get_control()
                velocities = self._clip_velocity(velocities)
                if self.env.mujoco_robot.name == "sawyer":
                    low_action = np.concatenate([velocities, action[7:]])
                elif self.env.mujoco_robot.name == "baxter":
                    low_action = np.concatenate([velocities, action[14:]])
                else:
                    raise Exception(
                        "Only Sawyer and Baxter robot environments are supported for IK "
                        "control currently."
                    )

        return ret

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        if self.use_abs_pose:
            return {
                "dpos": action[:3],
                # IK controller takes an absolute orientation in robot base frame
                "rotation": T.quat2mat(action[3:7]),
            }
        else:
            return {
                "dpos": action[:3],
                # IK controller takes an absolute orientation in robot base frame
                "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
            }

    def _clip_velocity(self, action):
        if self.clip_vel <= 0:
            return action
        for i in range(len(action)):
            action[i] = max(min(action[i], self.clip_vel), -self.clip_vel)
        return action
