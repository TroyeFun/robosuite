"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, MaterialModder, CameraModder


class DRWrapper(Wrapper):
    env = None

    def __init__(self, env, seed=None, path=None):
        super().__init__(env)
        self.action_noise = 1  # TODO: Should this be argument
        self.seed = seed
        self.tex_modder = TextureModder(self.env.sim, seed=seed, path=path)
        self.light_modder = LightingModder(self.env.sim, seed=seed)
        self.mat_modder = MaterialModder(self.env.sim, seed=seed)
        self.camera_modder =  CameraModder(sim=self.env.sim, camera_name=self.env.camera_name, seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.set_seed(seed)
        return super().reset()

    def set_seed(self, seed=None):
        self.seed = int(seed)
        self.tex_modder = TextureModder(self.env.sim, seed=self.seed)
        self.light_modder = LightingModder(self.env.sim, seed=self.seed)
        self.mat_modder = MaterialModder(self.env.sim, seed=self.seed)
        self.camera_modder = CameraModder(sim=self.env.sim, camera_name=self.env.camera_name, seed=self.seed)

    def step(self, action):
        #action += np.random.normal(scale=self.action_noise, size=action.shape)
        return super().step(action)

    def randomize_all(self):
        self.randomize_camera()
        self.randomize_texture()
        self.randomize_light()
        self.randomize_material()

    def randomize_texture(self):
        self.tex_modder.randomize()

    def randomize_light(self):
        self.light_modder.randomize()

    def randomize_material(self):
        self.mat_modder.randomize()

    def randomize_camera(self):
        self.camera_modder.randomize()
