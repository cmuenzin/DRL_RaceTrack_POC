import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon

def generate_sine_track(R=20.0, A=6.0, n_bulges=4, ds=0.2):
    thetas = np.arange(0, 2*np.pi, ds / R)
    pts = []
    for t in thetas:
        r = R + A * np.sin(n_bulges * t)
        pts.append((r * np.cos(t), r * np.sin(t)))
    pts.append(pts[0])
    return pts

class SimpleRaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # Fahrzeug
        self.car_width  = 1.0
        self.car_length = 1.5
        self.wheelbase  = 1.2
        self.velocity   = 2.0
        self.dt         = 0.1
        self.max_steer  = np.deg2rad(30)

        # Spaces
        self.action_space = spaces.Box(-self.max_steer, self.max_steer, (1,), np.float32)
        high = np.array([3*self.car_width, np.pi, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Track
        self.track_width = 3 * self.car_width
        self.center_line = generate_sine_track(R=20.0, A=6.0, n_bulges=4, ds=0.2)
        self.track_line  = LineString(self.center_line)
        self.outer_boundary = self.track_line.buffer(self.track_width/2, cap_style=2, join_style=2)
        self.inner_boundary = self.track_line.buffer(-self.track_width/2, cap_style=2, join_style=2)
        self.track_region   = self.outer_boundary.difference(self.inner_boundary)

        # Render
        self.render_mode = render_mode
        self.fig = None
        self.ax  = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = self.track_line.interpolate(0.0).coords[0]
        nxt = self.track_line.interpolate(1e-3)
        dx, dy = np.array(nxt.coords[0]) - np.array([self.x, self.y])
        self.phi = np.arctan2(dy, dx)
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        delta = float(np.clip(action, -self.max_steer, self.max_steer))
        self.phi += (self.velocity/self.wheelbase)*np.tan(delta)*self.dt
        self.x   += self.velocity*np.cos(self.phi)*self.dt
        self.y   += self.velocity*np.sin(self.phi)*self.dt

        pt = Point(self.x, self.y)
        if not self.track_region.contains(pt):
            self.done = True
            reward = -10.0
        else:
            self.done = False
            proj = self.track_line.project(pt)
            reward = proj / self.track_line.length

        obs = self._get_obs() if not self.done else np.zeros(3, dtype=np.float32)
        return obs, reward, self.done, False, {}

    def _get_obs(self):
        pt   = Point(self.x, self.y)
        proj = self.track_line.project(pt)
        progress = proj / self.track_line.length

        nearest = self.track_line.interpolate(proj)
        e_lat   = pt.distance(nearest)
        next_pt = self.track_line.interpolate(min(proj+1e-3, self.track_line.length))
        dx, dy  = np.array(next_pt.coords[0]) - np.array(nearest.coords[0])
        sign    = np.sign(dx*(self.y-nearest.y) - dy*(self.x-nearest.x))
        e_lat  *= sign

        track_phi = np.arctan2(dy, dx)
        e_head    = (self.phi - track_phi + np.pi) % (2*np.pi) - np.pi

        return np.array([e_lat, e_head, progress], dtype=np.float32)

    def render(self):
        if self.render_mode != 'human':
            return
        if self.fig is None:
            xs, ys = zip(*self.center_line)
            dx, dy = max(xs)-min(xs), max(ys)-min(ys)
            self.fig, self.ax = plt.subplots(figsize=(dx/10, dy/10))
        self.ax.clear()

        # Track-Füllung (optional)
        patch = MplPolygon(
            np.array(self.track_region.exterior.coords),
            facecolor='lightgray', edgecolor=None, alpha=0.5
        )
        self.ax.add_patch(patch)

        # Äußere Grenze
        xo, yo = self.outer_boundary.exterior.xy
        self.ax.plot(xo, yo, color='black', linewidth=2)
        # Innere Grenze
        xi, yi = self.inner_boundary.exterior.xy
        self.ax.plot(xi, yi, color='black', linewidth=2)

        # Auto
        car = Rectangle(
            (self.x - self.car_length/2, self.y - self.car_width/2),
            self.car_length, self.car_width,
            angle=np.degrees(self.phi), fc='red', ec='black'
        )
        self.ax.add_patch(car)

        self.ax.set_aspect('equal', 'box')
        self.ax.axis('off')
        plt.pause(1/self.metadata["render_fps"])

    def close(self):
        if self.fig:
            plt.close(self.fig)
