# envs/carla_parallel_parking_env.py

# Author: jones binadiegha Gabriel
import os
import time
import math
import numpy as np
import gym
from gym import spaces
import carla


# Utility: map [-1,1] → [0,1]
def _unit(x):
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))


# Geometry utils
def _yaw_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _rot_mat(yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _poly_contains(poly, pts):
    # poly: (N,2) in order; pts: (M,2)
    # Use winding/cross method for convex polygon (our bay rectangle is convex)
    def side(a, b, p):
        return np.cross(b - a, p - a)
    inside = []
    for p in pts:
        s = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i+1)%len(poly)]
            s.append(side(a, b, p))
        inside.append(np.all(np.array(s) >= -1e-4) or np.all(np.array(s) <= 1e-4))
        return np.array(inside)
    
class CarlaParallelParkingEnv(gym.Env):
    """Parallel parking in a single bay between two parked cars.
    Observations: {
    'image': uint8 (H,W,3), RGB camera
    'state': float32 (10,), [dx, dy, dist, yaw_err, speed, progress, collision_flag, t_norm, steer_norm, last_reward]
    }
    Actions (continuous, [-1,1]): [steer, throttle, brake]
    Success: ego fully inside target bay polygon, yaw aligned, speed ~ 0 for hold_steps.
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self,
            host=None,
            port=None,
            town="Town10HD_Opt",
            image_size=(96, 160),
            fov=90,
            max_steps=800,
            target_width=2.5,
            target_length=6.0,
            hold_steps=10,
            success_yaw_tol_deg=10.0,
            success_pos_tol=0.5,
            seed=0):
        super().__init__()
        self.host = host or os.getenv("CARLA_HOST", "10.140.13.244")
        self.port = int(port or os.getenv("CARLA_PORT", "2000"))
        self.town = town
        self.W, self.H = int(image_size[1]), int(image_size[0])
        self.fov = fov
        self.max_steps = max_steps
        self.target_w = target_width
        self.target_l = target_length
        self.hold_steps = hold_steps
        self.success_yaw_tol = math.radians(success_yaw_tol_deg)
        self.success_pos_tol = success_pos_tol
        self.rng = np.random.RandomState(seed)

        # Gym spaces
        self.observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8),
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


        # CARLA client state
        self.client = None
        self.world = None
        self.map = None
        self.synchronous = True
        self.delta_seconds = 1/20.0
        self.ego = None
        self.sensors = []
        self.image_np = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.collision = 0
        self.steps = 0
        self.last_reward = 0.0
        self.held_success = 0
        self._setup_client()

        # --------------- CARLA boilerplate ---------------
        def _setup_client(self):
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()
            if self.world.get_map().name != self.town:
                self.world = self.client.load_world(self.town)
            self.map = self.world.get_map()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.delta_seconds
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)


        def _spawn_actor(self, blueprint_filter, transform, role_name=None):
            bp_lib = self.world.get_blueprint_library()
            bps = bp_lib.filter(blueprint_filter)
            bp = bps[0]
            if role_name:
                bp.set_attribute("role_name", role_name)
            actor = self.world.spawn_actor(bp, transform)
            return actor
        
        def _destroy_actors(self, actors):
            for a in actors:
                try:
                    a.destroy()
                except Exception:
                    pass

    # --------------- Scenario layout ---------------
    def _choose_bay(self):
        # Choose a road segment with sidewalk; hardcode a known location in Town10HD_Opt
        # Reference transform (manually chosen good curb segment)
        # You can refine these Transforms later.
        base = carla.Transform(carla.Location(x=50.0, y=-10.0, z=0.1), carla.Rotation(yaw=0.0))


        curb_dir = np.array([1.0, 0.0], dtype=np.float32) # along +x
        curb_right = np.array([0.0, -1.0], dtype=np.float32) # towards -y


        # Parking bay center relative to base
        bay_center = np.array([0.0, 0.0], dtype=np.float32) # at base for simplicity
        bay_yaw = 0.0 # parallel to curb


        # Parked cars positions (front and rear of the gap)
        gap = self.target_l
        car_len = 4.6
        buffer = 0.5
        front_offset = (gap/2 + car_len/2 + buffer) * curb_dir
        rear_offset = -(gap/2 + car_len/2 + buffer) * curb_dir
        side_offset = 1.6 * curb_right # offset from curb into lane


        front_loc = bay_center + front_offset + side_offset
        rear_loc = bay_center + rear_offset + side_offset


        # Ego spawn a bit behind and out from the bay
        ego_offset = (-10.0 * curb_dir) + (3.2 * curb_right)
        ego_loc = bay_center + ego_offset


        def to_transform(v2, yaw_deg):
            return carla.Transform(
                carla.Location(x=base.location.x + float(v2[0]),
                    y=base.location.y + float(v2[1]),
                    z=0.1),
                carla.Rotation(yaw=base.rotation.yaw + yaw_deg))
        return {
            "bay_center": bay_center,
            "bay_yaw": bay_yaw,
            "front_tr": to_transform(front_loc, 0.0),
            "rear_tr": to_transform(rear_loc, 0.0),
            "ego_tr": to_transform(ego_loc, 0.0),
            "base": base,
            }
    def _target_polygon_world(self, layout):
        # Rectangle centered at bay, aligned with bay_yaw
        cx, cy = layout["bay_center"]
        yaw = layout["bay_yaw"]
        R = _rot_mat(yaw)
        dx, dy = self.target_l/2.0, self.target_w/2.0
        corners_local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
        corners = (corners_local @ R.T) + np.array([cx, cy])
        # to world (Town origin shift by base transform)
        corners_world = corners + np.array([layout["base"].location.x, layout["base"].location.y])
        return corners_world
    
    # --------------- Sensors ---------------
    def _attach_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.W))
        cam_bp.set_attribute("image_size_y", str(self.H))
        cam_bp.set_attribute("fov", str(self.fov))
        cam_loc = carla.Location(x=1.2, z=1.4)
        cam_rot = carla.Rotation(pitch=0.0)
        cam = self.world.spawn_actor(cam_bp, carla.Transform(cam_loc, cam_rot), attach_to=self.ego)


        def _on_img(img):
            array = np.frombuffer(img.raw_data, dtype=np.uint8)
            array = array.reshape((img.height, img.width, 4))[:, :, :3]
            self.image_np = array
            cam.listen(_on_img)


            col_bp = bp_lib.find("sensor.other.collision")
            col = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego)
            def _on_col(event):
                self.collision += 1
            col.listen(_on_col)


            self.sensors = [cam, col]
    # --------------- Reset ---------------
    def reset(self):
        self._cleanup_episode()
        self.steps = 0
        self.collision = 0
        self.last_reward = 0.0
        self.held_success = 0


        # Clear previous actors if any lingering
        self.client.get_world().tick()


        layout = self._choose_bay()
        self.layout = layout


        # Spawn parked cars
        self.parked = []
        for tr in [layout["front_tr"], layout["rear_tr"]]:
            car = self._spawn_actor("vehicle.tesla.model3", tr, role_name="parked")
            car.set_autopilot(False)
            self.parked.append(car)


        # Spawn ego
        self.ego = self._spawn_actor("vehicle.tesla.model3", layout["ego_tr"], role_name="ego")
        self.ego.set_autopilot(False)


        # Sensors
        self._attach_sensors()


        # Warm up a few ticks to fill first image
        for _ in range(5):
            self.world.tick()


        obs = self._get_obs()
        return obs
    def _cleanup_episode(self):
        if self.sensors:
            for s in self.sensors:
                try:
                    s.stop()
                except Exception:
                    pass
            self._destroy_actors(self.sensors)
            self.sensors = []
        if getattr(self, "ego", None):
            self._destroy_actors([self.ego])
            self.ego = None
        if getattr(self, "parked", None):
            self._destroy_actors(self.parked)
            self.parked = []
    # --------------- Step / Obs / Reward ---------------
    def step(self, action):
        self.steps += 1
        steer, thr, brk = float(action[0]), float(action[1]), float(action[2])
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = _unit(thr)
        control.brake = _unit(brk)
        control.reverse = False
        self.ego.apply_control(control)


        self.world.tick()


        obs = self._get_obs()
        reward, done, info = self._compute_reward_done()
        self.last_reward = reward
        return obs, reward, done, info

    def _ego_state(self):
        tf = self.ego.get_transform()
        v = self.ego.get_velocity()
        speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
        yaw = math.radians(tf.rotation.yaw)
        pos = np.array([tf.location.x, tf.location.y], dtype=np.float32)
        return pos, yaw, speed


    def _get_obs(self):
        pos, yaw, speed = self._ego_state()
        # Target polygon center/yaw
        cx, cy = self.layout["bay_center"] + np.array([self.layout["base"].location.x, self.layout["base"].location.y])
        target_center = np.array([cx, cy])
        dxdy = target_center - pos
        dist = float(np.linalg.norm(dxdy))
        yaw_err = float(_yaw_diff(yaw, self.layout["bay_yaw"]))
        t_norm = self.steps / float(self.max_steps)
        steer_norm = 0.0 # not tracked here; can feed last command if desired
        state = np.array([
        dxdy[0], dxdy[1], dist, yaw_err, speed, 0.0, float(self.collision>0), t_norm, steer_norm, self.last_reward
        ], dtype=np.float32)
        return {"image": self.image_np.copy(), "state": state}
    
    def _compute_reward_done(self):
        pos, yaw, speed = self._ego_state()
        poly = self._target_polygon_world(self.layout)
        # Ego bbox corners (approx): use car dims ~ (L=4.6,W=1.9)
        L, Wd = 4.6, 1.9
        R = _rot_mat(yaw)
        corners_local = np.array([[-L/2, -Wd/2], [L/2, -Wd/2], [L/2, Wd/2], [-L/2, Wd/2]], dtype=np.float32)
        ego_corners = (corners_local @ R.T) + pos
        inside = _poly_contains(poly, ego_corners)
        in_box = bool(np.all(inside))


        cx, cy = self.layout["bay_center"] + np.array([self.layout["base"].location.x, self.layout["base"].location.y])
        target_center = np.array([cx, cy])
        dist_center = float(np.linalg.norm(target_center - pos))
        yaw_err = abs(_yaw_diff(yaw, self.layout["bay_yaw"]))


        # Reward shaping
        r_dist = -0.5 * dist_center
        r_yaw = -0.2 * (yaw_err)
        r_speed = -0.05 * max(0.0, speed - 0.2)
        r_col = -5.0 if self.collision else 0.0
        r_time = -0.001
        reward = r_dist + r_yaw + r_speed + r_col + r_time


        done = False
        success = False
        if in_box and yaw_err < self.success_yaw_tol and speed < 0.1:
            self.held_success += 1
            reward += 1.0 # holding bonus
        if self.held_success >= self.hold_steps:
            reward += 50.0
            done = True
            success = True
        else:
            self.held_success = 0


        if self.collision:
            done = True
            reward -= 20.0

        if self.steps >= self.max_steps:
            done = True


        info = {"success": success, "in_box": in_box,
        "dist_center": dist_center, "yaw_err": yaw_err,
        "collisions": int(self.collision)}
        return float(reward), bool(done), info

    def render(self, mode="rgb_array"):
        return self.image_np.copy()
    
    def close(self):
        self._cleanup_episode()
        if self.world is not None:
            # Restore async mode just in case
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception:
                pass