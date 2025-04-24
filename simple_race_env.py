# simple_race_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

class SimpleRaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # === Fahrzeugparameter ===
        self.car_width = 1.0
        self.wheelbase = 2.0
        self.velocity = 2.0    # konstante Geschwindigkeit
        self.dt = 0.1          # Zeitschritt
        self.max_steer = np.deg2rad(30)

        # === Action- & Observation-Space ===
        self.action_space = spaces.Box(
            low=-self.max_steer, high=self.max_steer,
            shape=(1,), dtype=np.float32
        )
        high = np.array([3*self.car_width, np.pi, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )

        # === Track-Generierung ===
        self.track_width = 3 * self.car_width
        self.center_line = self._generate_track()
        self.track_line = LineString(self.center_line)
        self.track_region = self.track_line.buffer(
            self.track_width/2, cap_style=2, join_style=2
        )

        # === Render-Setup ===
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

    def _generate_track(self):
        # Einfache Kreis-Strecke als PoC
        R = 20.0
        n_points = 360
        thetas = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return [(R*np.cos(t), R*np.sin(t)) for t in thetas]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Startposition = erster Punkt der Mittellinie
        self.x, self.y = self.center_line[0]
        self.phi = 0.0
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        # 1) Kinematik-Update (Bicycle Model)
        delta = float(np.clip(action, -self.max_steer, self.max_steer))
        self.phi += (self.velocity/self.wheelbase) * np.tan(delta) * self.dt
        self.x += self.velocity * np.cos(self.phi) * self.dt
        self.y += self.velocity * np.sin(self.phi) * self.dt

        # 2) Collision/Boundary-Check
        point = Point(self.x, self.y)
        if not self.track_region.contains(point):
            self.done = True
            reward = -10.0
        else:
            self.done = False
            # Fortschritt (0..1) entlang Mittellinie
            proj = self.track_line.project(point)
            progress = proj / self.track_line.length

            # lateraler Fehler
            nearest = self.track_line.interpolate(proj)
            e_lat = point.distance(nearest)
            # Vorzeichenbestimmung
            next_pt = self.track_line.interpolate(proj + 1e-3)
            dx, dy = np.array(next_pt.coords[0]) - np.array(nearest.coords[0])
            sign = np.sign(dx*(self.y-nearest.y) - dy*(self.x-nearest.x))
            e_lat *= sign

            # heading error
            track_phi = np.arctan2(dy, dx)
            e_head = (self.phi - track_phi + np.pi) % (2*np.pi) - np.pi

            reward = progress

        obs = self._get_obs() if not self.done else np.zeros(3, dtype=np.float32)
        return obs, reward, self.done, False, {}

    def _get_obs(self):
        # Erneute Berechnung von [e_lat, e_head, progress]
        point = Point(self.x, self.y)
        proj = self.track_line.project(point)
        progress = proj / self.track_line.length

        nearest = self.track_line.interpolate(proj)
        e_lat = point.distance(nearest)
        next_pt = self.track_line.interpolate(proj + 1e-3)
        dx, dy = np.array(next_pt.coords[0]) - np.array(nearest.coords[0])
        sign = np.sign(dx*(self.y-nearest.y) - dy*(self.x-nearest.x))
        e_lat *= sign

        track_phi = np.arctan2(dy, dx)
        e_head = (self.phi - track_phi + np.pi) % (2*np.pi) - np.pi

        return np.array([e_lat, e_head, progress], dtype=np.float32)

    def render(self):
        if self.render_mode != 'human':
            return
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.clear()
        # Mittellinie
        xs, ys = zip(*self.center_line)
        self.ax.plot(xs, ys, color='gray')
        # Boundary
        x_r, y_r = self.track_region.exterior.xy
        self.ax.plot(x_r, y_r, color='black')
        # Auto-Pfeil
        self.ax.arrow(
            self.x, self.y,
            np.cos(self.phi), np.sin(self.phi),
            head_width=0.5, head_length=0.7, fc='red'
        )
        self.ax.set_aspect('equal', 'box')
        plt.pause(1/self.metadata['render_fps'])

    def close(self):
        if self.fig:
            plt.close(self.fig)
