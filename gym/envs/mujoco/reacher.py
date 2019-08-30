import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.step_violation = 0
        self.total_violation = 0
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        pre_ob = self._get_obs()
        reward = - np.linalg.norm(vec)
        if(pre_ob[1] > 0):
            reward = reward - 100/abs(pre_ob[0])
        if(pre_ob[1] > 0 and abs(pre_ob[0]) < 0.09):
            self.step_violation += 1
            self.total_violation += 1
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.step_violation = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[0] = np.pi/6 + self.np_random.uniform(low=-0.1, high=0.1)
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        self.goal[0] = np.random.uniform(-0.2,0)
        self.goal[1] = np.random.uniform(-0.1,0)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            self.get_body_com("fingertip"),
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def cost_fn(self,states, actions, next_states):
        return np.square(next_states[:,0]) + np.square(next_states[:,1] + 0.2) 

    def cost_fn_des(self,states, actions, next_states):
        return np.square(next_states[:,-1]) + np.square(next_states[:,-2]) 

    def choice_cost_func(self,states, actions, next_states):
        safe_cost = 0
        if(next_states[1] > 0):
            safe_cost = 100/abs(next_states[0])
        free_cost = np.square(next_states[-1]) + np.square(next_states[-2])
        return safe_cost + free_cost
