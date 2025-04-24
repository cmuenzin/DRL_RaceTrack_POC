# main.py
import time
import numpy as np
from simple_race_env import SimpleRaceEnv

def run_random_rollout(env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Abgebrochen bei Schritt {step+1}, Reward={total_reward:.2f}")
            break
    else:
        print(f"Maximale Schritte erreicht, Reward={total_reward:.2f}")
    time.sleep(1)  # kurze Pause vor dem nächsten Rollout

def main():
    env = SimpleRaceEnv(render_mode='human')
    for episode in range(3):
        print(f"=== Episode {episode+1} ===")
        run_random_rollout(env)
    env.close()

if __name__ == '__main__':
    main()
