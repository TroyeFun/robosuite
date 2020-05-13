from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import BinsArena
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
)
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import PickPlaceTask, UniformRandomSampler
import robosuite.utils.visualize as vis
from robosuite.utils.visualize import color_rgba

from ipdb import set_trace as pdb

class SawyerPickPlaceMultiTask(SawyerEnv):
    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        multi_task_mode=True,
        #reset_color=False,
        reset_color=True,
        with_target=False,
        place_at_center=True,
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with all types of objects.

                1: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            object_type (string): if provided, should be one of "milk", "bread", "cereal",
                or "can". Determines which type of object will be spawned on every
                environment reset. Only used if @single_object_mode is 2.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that 
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            multi_task_mode (bool): True if divide the whole task into two subtask (pick, place).

            reset_color (bool): True if remove the texture of objects and reset their color

            with_target (bool): True if set a specific target object
        """

        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        if object_type is not None:
            assert (
                object_type in self.object_to_id.keys()
            ), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.target_id = self.object_to_id[
                object_type
            ]  # use for convenient indexing
        self.target_object = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reset color
        self.reset_color = reset_color

        # multitask mode
        self.multi_task_mode = multi_task_mode
        self.current_task = None

        # set target object
        self.with_target = with_target

        self.place_at_center = place_at_center

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = BinsArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.56, 0, 0])

        self.ob_inits = [MilkObject, BreadObject, CerealObject, CanObject]
        self.vis_inits = [
            MilkVisualObject,
            BreadVisualObject,
            CerealVisualObject,
            CanVisualObject,
        ]
        self.item_names = ["Milk", "Bread", "Cereal", "Can"]
        self.item_names_org = list(self.item_names)

        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]), self.vis_inits[j]()))
        self.visual_objects = lst

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]) + "0", ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = PickPlaceTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
        )

        self.target_object = (self.item_names[0] + "{}").format(0)
        if self.single_object_mode == 1:
            self.target_object = (random.choice(self.item_names) + "{}").format(0)
        elif self.single_object_mode == 2:
            self.target_object = (self.item_names[self.object_id] + "{}").format(0)

        # warning set place range
        if self.place_at_center and (self.single_object_mode == 1 or self.single_object_mode == 2):
            self.model.place_objects(place_radius=0.1, obj_names=[self.target_object])
        else:
            self.model.place_objects()

        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin2_body.get("pos"))
        self.bin_size = self.model.bin_size

        if self.reset_color:
            self._setup_color()

    def _setup_color(self):
        colors = ['purple', 'green', 'blue', 'yellow']
        self.object_color = OrderedDict(zip(self.mujoco_objects.keys(), colors))
        for obj_name, color in self.object_color.items():
            ## set after model loaded
            #geom_id = self.sim.model.geom_name2id(obj)
            #self.sim.model.geom_rgba[geom_id, :] = color_rgba[color]
            #self.sim.model.geom_matid[geom_id] = -1   # unset material

            # set before model loaded by MjSim
            #obj_visual = self.mujoco_objects[obj_name].worldbody.find("body/body[@name='visual']")
            #geoms = obj_visual.findall('geom')  # include 2 groups
            ## TODO: decide which group to set
            #for geom in geoms:
            #    geom.set('material_tmp', geom.get('material'))
            #    geom.attrib.pop('material')
            #    geom.set('rgba', ' '.join(map(str, color_rgba[color])))
            
            geom = self.model.worldbody.find("body[@name='{}']/geom".format(obj_name))
            geom.attrib.pop('material')
            geom.set('rgba', ' '.join(map(str, color_rgba[color])))

    def clear_objects(self, obj):
        """
        Clears objects with name @obj out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.
        """
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name)[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)

        # for checking distance to / contact with objects we want to pick up
        self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.ob_inits))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.ob_inits), 3))
        for j in range(len(self.ob_inits)):
            bin_id = j
            bin_x_low = self.bin_pos[0]
            bin_y_low = self.bin_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[j, :] = [bin_x_low, bin_y_low, self.bin_pos[2]]

    def _reset_internal(self):
        super()._reset_internal()

        # reset positions of objects, and move objects out of the scene depending on the mode
        if self.single_object_mode == 1 or self.single_object_mode == 2:
            self.clear_objects(self.target_object)

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        if self.multi_task_mode:
            self.current_task = 'pick'

        if self.with_target:
            self._set_target_object()

    def _set_target_object(self):
        # TODO: higher level strategy for target choice.

        # choose by random
        if self.single_object_mode == 0:
            self.target_object = (random.choice(self.item_names) + "{}").format(0)

        if self.reset_color:
            #self.target_color = vis.color2id[self.object_color[self.target_object]]
            self.target_color = self.object_color[self.target_object]

        self.target_id = self.object_to_id[self.target_object.strip('0').lower()]
        self.target_body_id = self.obj_body_id[self.target_object]
        self.target_geom_id = self.obj_geom_id[self.target_object]

    def reward(self, action=None):
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        if self.multi_task_mode:
            self._update_current_task()
            reward += self._reward_with_target()
        # add in shaped rewards
        elif self.reward_shaping: 
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        objs_to_reach = []
        geoms_to_grasp = []
        target_bin_placements = []
        for i in range(len(self.ob_inits)):
            if self.objects_in_bins[i]:
                continue
            obj_str = str(self.item_names[i]) + "0"
            objs_to_reach.append(self.obj_body_id[obj_str])
            geoms_to_grasp.append(self.obj_geom_id[obj_str])
            target_bin_placements.append(self.target_bin_placements[i])
        target_bin_placements = np.array(target_bin_placements)

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # get reaching reward via minimum distance to a target object
            target_object_pos = self.sim.data.body_xpos[objs_to_reach]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dists = np.linalg.norm(
                target_object_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom1)
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom2)
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.bin_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        ### hover reward for getting object above bin ###
        r_hover = 0.
        if len(objs_to_reach):
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
            y_check = (
                np.abs(object_xy_locs[:, 1] - target_bin_placements[:, 1])
                < self.bin_size[1] / 4.
            )
            x_check = (
                np.abs(object_xy_locs[:, 0] - target_bin_placements[:, 0])
                < self.bin_size[0] / 4.
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(
                target_bin_placements[:, :2] - object_xy_locs, axis=1
            )
            # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(objs_to_reach))
            r_hover_all[objects_above_bins] = lift_mult + (
                1 - np.tanh(10.0 * dists[objects_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover_all[objects_not_above_bins] = r_lift + (
                1 - np.tanh(10.0 * dists[objects_not_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin_pos[0]
        bin_y_low = self.bin_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            obj_pos[2] > self.bin_pos[2]
            and obj_pos[0] < bin_x_high
            and obj_pos[0] > bin_x_low
            and obj_pos[1] < bin_y_high
            and obj_pos[1] > bin_y_low
            and obj_pos[2] < self.bin_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for i in range(len(self.item_names_org)):

                if self.single_object_mode == 2 and self.target_id != i:
                    # Skip adding to observations
                    continue

                obj_str = str(self.item_names_org[i]) + "0"
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

                # get relative pose of object in gripper frame
                object_pose = T.pose2mat((obj_pos, obj_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(obj_str)] = rel_pos
                di["{}_to_eef_quat".format(obj_str)] = rel_quat

                object_state_keys.append("{}_pos".format(obj_str))
                object_state_keys.append("{}_quat".format(obj_str))
                object_state_keys.append("{}_to_eef_pos".format(obj_str))
                object_state_keys.append("{}_to_eef_quat".format(obj_str))

            if self.single_object_mode == 1:
                # Zero out other objects observations
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.target_object:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_eef_pos".format(obj_str)] *= 0.0
                        di["{}_to_eef_quat".format(obj_str)] *= 0.0

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            
            # all in bin, but why r_reach < 0.6 (dist > 0.042)?
            self.objects_in_bins[i] = int(
                (not self.not_in_bin(obj_pos, i)) and r_reach < 0.6  
            )

        # returns True if a single object is in the correct bin
        if self.single_object_mode == 1 or self.single_object_mode == 2:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.ob_inits)

    def _update_current_task(self):
        #return

        if not self._check_picked():
            self.current_task == 'pick'
        else:
            self.current_task = 'place'
        #TODO set target to another object if the current target is placed

    def _check_picked(self):
        """
        Returns True if target object has been picked and lift.
        """
        target_height = self.sim.data.body_xpos[self.target_body_id][2]
        table_height = self.table_full_size[2]

        # target is higher than the table top above a margin
        return target_height > table_height + 0.1

    def _check_placed(self):
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.target_body_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = 1 - np.tanh(10.0 * dist)
        object_in_bin = int(
            (not self.not_in_bin(obj_pos, self.target_id)) #and r_reach < 0.6
        )

        return object_in_bin > 0

    def _reward_with_target(self):
        #TODO
        if self.current_task == 'pick':
            return self._reward_pick()
        elif self.current_task == 'place':
            return self._reward_place()
        else:
            raise NotImplementedError

    def _reward_without_target(self):
        #TODO
        pass

    def _reward_pick(self):
        #TODO targetless mode
        # support target mode only
        # reaching reward
        reward = self._check_picked()

        target_pos = self.sim.data.body_xpos[self.target_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - target_pos)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        reward += reaching_reward

        # grasping reward
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.target_geom_id:
                touch_left_finger = True
            if c.geom1 == self.target_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.target_geom_id:
                touch_right_finger = True
            if c.geom1 == self.target_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        if touch_left_finger and touch_right_finger:
            reward += 0.25
        return reward

    def _reward_place(self):
        #TODO
        # notice to set picked object before calculating reward
        # notice reward for target object dropped
        # support target mode only
        
        ### hover reward for getting object above bin ###
        base_reward = 2.25  # max reward of picking
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7
        place_mult = 10

        reward = base_reward + self._check_placed() * place_mult
        objs_to_reach = [self.target_body_id]
        target_bin_placements = np.array([self.target_bin_placements[self.target_id]])

        ### lifting reward for picking up an object ###
        r_lift = 0.
        z_target = self.bin_pos[2] + 0.25
        object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
        z_dists = np.maximum(z_target - object_z_locs, 0.)
        r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
            lift_mult - grasp_mult
            )

        # segment objects into left of the bins and above the bins
        object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
        y_check = (
            np.abs(object_xy_locs[:, 1] - target_bin_placements[:, 1])
            < self.bin_size[1] / 4.
        )
        x_check = (
            np.abs(object_xy_locs[:, 0] - target_bin_placements[:, 0])
            < self.bin_size[0] / 4.
        )
        objects_above_bins = np.logical_and(x_check, y_check)
        objects_not_above_bins = np.logical_not(objects_above_bins)
        dists = np.linalg.norm(
            target_bin_placements[:, :2] - object_xy_locs, axis=1
        )
        # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
        r_hover_all = np.zeros(len(objs_to_reach))
        r_hover_all[objects_above_bins] = lift_mult + (  # not reward lift if obj on the right of bin
            1 - np.tanh(10.0 * dists[objects_above_bins])
        ) * (hover_mult - lift_mult)
        r_hover_all[objects_not_above_bins] = r_lift + (  # reward lift if obj on the left of bin
            1 - np.tanh(10.0 * dists[objects_not_above_bins])
        ) * (hover_mult - lift_mult)
        r_hover = np.max(r_hover_all)

        reward += r_hover
        return reward


    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba

"""
Without target object
    Single-task 
    see sawyer_pick_place.py
"""

"""
Without target object
    Multi-task
"""

class SawyerPickPlaceSingleMultiTask(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=1, **kwargs)

class SawyerPickPlaceMilkMultiTask(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="milk", **kwargs)

class SawyerPickPlaceBreadMultiTask(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)

class SawyerPickPlaceCerealMultiTask(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="cereal", **kwargs)

class SawyerPickPlaceCanMultiTask(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="can", **kwargs)

"""
With target object
    Single Task
"""
class SawyerPickPlaceTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=0, multi_task_mode=False, with_target=True, **kwargs)

class SawyerPickPlaceSingleTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=1, multi_task_mode=False, with_target=True, **kwargs)

class SawyerPickPlaceMilkTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="milk", multi_task_mode=False, with_target=True, **kwargs)

class SawyerPickPlaceBreadTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="bread", multi_task_mode=False, with_target=True, **kwargs)

class SawyerPickPlaceCerealTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="cereal", multi_task_mode=False, with_target=True, **kwargs)

class SawyerPickPlaceCanTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, object_type="can", multi_task_mode=False, with_target=True, **kwargs)

"""
With target object
    Multi task
"""

class SawyerPickPlaceMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=0, with_target=True, **kwargs)


class SawyerPickPlaceSingleMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=1, with_target=True, **kwargs)


class SawyerPickPlaceMilkMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, with_target=True, object_type="milk", **kwargs)


class SawyerPickPlaceBreadMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, with_target=True, object_type="bread", **kwargs)


class SawyerPickPlaceCerealMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, with_target=True, object_type="cereal", **kwargs)


class SawyerPickPlaceCanMultiTaskTarget(SawyerPickPlaceMultiTask):
    def __init__(self, **kwargs):
        exclude_keys = ['single_object_mode', 'object_type', 'multi_task_mode', 'with_target']
        for key in exclude_keys:
            assert key not in kwargs, 'invalid set of argument: ' + key
        super().__init__(single_object_mode=2, with_target=True, object_type="can", **kwargs)

if __name__ == '__main__':
    import ipdb
    import cv2
    import os
    import pcl
    import math


    env = SawyerPickPlaceSingleMultiTaskTarget(has_renderer=True,
    #env = SawyerPickPlace(has_renderer=True,
                     camera_depth=True,
                     #camera_name='birdview')
                     #camera_name='frontview')
                     camera_name='agentview')


    #env.sim.model.geom_matid[67:71] = -1   # set material to -1
    objs = ['Milk0', 'Can0', 'Bread0', 'Cereal0']
    color_types = ['blue', 'green', 'purple', 'yellow']
    ids = dict([(obj,env.sim.model.geom_name2id(obj)) for obj in objs])
    obj_colors = dict(zip(objs, color_types))
    
    rgba_color = {
        'blue':  [0, 0, 3, 1],  # 120,255,255
        'green': [0, 3, 0, 1],  # 60,255,255
        'red':   [3, 0, 0, 1], # 0, 255,255
        'yellow':[3, 3, 0, 1],  # 30,255,255
        'purple':[3, 0, 3, 1],  # 150,255,255
    }
    hsv_range = {
        'blue':   [[115,150,150],[125,255,255]],  # 120,255,255
        'green':  [[55 ,150,150],[65 ,255,255]],  # 60,255,255
        'red':    [[0  ,150,150],[10 ,255,255]],  # 0, 255,255
        'yellow': [[25 ,150,150],[35 ,255,255]],  # 30,255,255
        'purple': [[145,150,150],[155,255,255]],  # 150,255,255

    }

    #for obj in objs:
    #    env.sim.model.geom_rgba[ids[obj],:] = rgba_color[obj_colors[obj]]

    color_type = 'yellow'

    """
        while True:
            env.reset()
            print(env.current_task)
        exit()

    """
    while True:
        #env.sim.model.geom_matid[67:71] = -1   # set material to -1
        #for obj in objs:
        #    env.sim.model.geom_rgba[ids[obj],:] = rgba_color[obj_colors[obj]]
        env.render()
        obs = env._get_observation()
        color, depth = obs['image'], obs['depth']
        color =cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color = cv2.flip(color, 0) # horizontal flip
        depth = cv2.flip(depth, 0) # horizontal flip
        cv2.imshow('color', color)
        cv2.waitKey(100)
        #cv2.imshow('depth', np.tile(depth[:,:,np.newaxis], (1,1,3)))
        #cv2.waitKey(100)

        #lower, upper = np.array(hsv_range[color_type])
        #hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(hsv, lower, upper)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(100)

        #x = np.arange(env.camera_width) - env.camera_width / 2
        #x = np.tile(x, (env.camera_height, 1))
        #y = np.arange(env.camera_height)[:, np.newaxis] - env.camera_height / 2
        #y = np.tile(y, (1, env.camera_width))
        #
        #cam_id = env.sim.model.camera_name2id(env.camera_name)
        #fovy = env.sim.model.cam_fovy[cam_id]
        #cam_pos = env.sim.model.cam_pos[cam_id]
        #cam_quat = env.sim.model.cam_quat[cam_id]
        #cam_mat0 = T.pose2mat((cam_pos, T.convert_quat(cam_quat, 'xyzw')))
        ## can directly get from 
        #cam_mat = env.sim.model.cam_mat0[cam_id].reshape(3,3)
        #cam_pos = cam_pos.reshape(3,1)
        #
        #f = 0.5 * env.camera_height / math.tan(fovy * math.pi / 360)

        ##f *= 0.1
        ##depth *= 2.7

        #x = x * depth / f
        #y = y * depth / f

        #obj_index = mask > 0
        #obj_x = x[obj_index][np.newaxis,:]
        #obj_y = -y[obj_index][np.newaxis,:]
        #obj_z = -depth[obj_index][np.newaxis,:]
        #obj_points = np.concatenate((obj_x, obj_y, obj_z), axis=0)

        ## transform
        #obj_points = cam_mat.dot(obj_points) + cam_pos
        #obj_points = obj_points.T


        #colors = np.ones((obj_points.shape[0], 1)) * 255
        #points = np.concatenate((obj_points, colors), axis=1).astype('float32')
        #cloud = pcl.PointCloud_PointXYZRGB(points)
        #pcl.save(cloud, '../../exp/cloud1.pcd')

        #import utils.visualize as vis
        #rgbd_img = np.concatenate([obs['image'].transpose(2,0,1), depth[np.newaxis,:,:]])
        #pcd = vis.get_pcd(rgbd_img, cam_mat, cam_pos, f, 'yellow')
        #vis.save_pcd(pcd, '../../exp/')
        #

        ipdb.set_trace()
