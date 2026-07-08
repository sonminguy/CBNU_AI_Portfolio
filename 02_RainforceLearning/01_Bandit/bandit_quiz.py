import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


def run_bandit(runs, steps, epsilon):
    all_rates = np.zeros((runs, steps))
    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        
        all_rates[run] = rates
    return np.average(all_rates, axis=0)

avg_rates_1 = run_bandit(200, 1000, 0.1)
avg_rates_2 = run_bandit(200, 1000, 0.3)
avg_rates_3 = run_bandit(200, 1000, 0.01)


plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates_1)
plt.plot(avg_rates_2)
plt.plot(avg_rates_3)
plt.legend(["0.1", "0.3", "0.01"])
plt.show()