import time
import numpy as np
from simple_race_env import SimpleRaceEnv

NUM_EPISODES = 100
DISPLAY_EVERY = 10
MAX_STEPS = 500


def run_random_rollout(env, max_steps=MAX_STEPS):
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f" Abgebrochen bei Schritt {step+1}, Reward={total_reward:.2f}")
            break
    else:
        print(f" Maximale Schritte erreicht, Reward={total_reward:.2f}")
    time.sleep(0.5)


def main():
    env = SimpleRaceEnv(render_mode='human')
    for ep in range(1, NUM_EPISODES+1):
        if ep % DISPLAY_EVERY == 0:
            print(f"=== Episode {ep} ===")
            run_random_rollout(env)
        else:
            # Schnelle Simulation ohne Rendering
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, _, _ = env.step(action)
    env.close()

if __name__ == '__main__':
    main()
