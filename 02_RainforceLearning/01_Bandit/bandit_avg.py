# Multi-Armed Bandit 시뮬레이션 스크립트
# epsilon-greedy 전략을 사용한 평균 보상률 분석

import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent

# 시뮬레이션 파라미터 설정
runs = 200      # 독립적인 실험 반복 횟수
steps = 1000    # 각 실험당 스텝 수
epsilon = 0.1   # epsilon-greedy 전략의 탐험 확률 (10%)

# 모든 실행의 보상률을 저장할 배열 초기화
all_rates = np.zeros((runs, steps))

# 여러 번의 독립적인 실험 수행
for run in range(runs):
    # 새로운 Bandit 환경과 Agent 생성
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0  # 누적 보상
    rates = []        # 각 스텝별 평균 보상률

    # 각 실험마다 steps만큼 반복
    for step in range(steps):
        # Agent가 행동 선택 (epsilon-greedy)
        action = agent.get_action()
        
        # 선택한 행동으로 Bandit에서 보상 획득
        reward = bandit.play(action)
        
        # 획득한 보상으로 Agent의 가치 추정 업데이트
        agent.update(action, reward)
        
        # 누적 보상 갱신
        total_reward += reward
        
        # 현재 스텝까지의 평균 보상률 계산 및 저장
        rates.append(total_reward / (step + 1))

    # 현재 실행의 보상률을 전체 배열에 저장
    all_rates[run] = rates

# 모든 실행에 대한 평균 보상률 계산 (축 0에 대해 평균)
avg_rates = np.average(all_rates, axis=0)

# 그래프 레이블 설정
plt.ylabel("Rates")   # Y축: 평균 보상률
plt.xlabel("Steps")   # X축: 스텝 수

# 평균 보상률 그래프 그리기
plt.plot(avg_rates)
plt.show()