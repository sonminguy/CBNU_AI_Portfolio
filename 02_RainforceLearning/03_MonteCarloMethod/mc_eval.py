from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9  # 할인율
        self.action_size = 4  # 행동 공간 크기
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 행동 확률 (균등 분포)
        self.pi = defaultdict(lambda: random_actions)  # 행동 정책 (상태마다 행동 확률)
        self.V = defaultdict(lambda: 0)  # 상태 가치 함수 (초기값 0)
        self.cnts = defaultdict(lambda: 0)  # 상태 방문 횟수 (초기값 0)
        self.memory = []  # 에피소드 동안의 경험 저장

    def get_action(self, state):
        action_probs = self.pi[state]  # 현재 상태에서 행동 확률 가져오기
        actions = list(action_probs.keys())  # 행동 목록
        probs = list(action_probs.values())  # 행동 확률 목록
        return np.random.choice(actions, p=probs)  # 행동 확률에 따라 행동 선택
    
    def add(self, state, action, reward):
        self.memory.append((state, action, reward))  # 경험 저장

    def reset(self):
        self.memory.clear()  # 경험 초기화

    def eval(self):
        G = 0  # 누적 보상 초기화
        for state, action, reward in reversed(self.memory):  # 에피소드 경험을 역순으로 처리
            G = reward + self.gamma * G  # 누적 보상 계산
            self.cnts[state] += 1  # 상태 방문 횟수 증가
            self.V[state] += (G - self.V[state]) / self.cnts[state]  # 상태 가치 함수 업데이트


if __name__ == "__main__":
    env = GridWorld()  # 그리드 월드 환경 생성
    agent = RandomAgent()  # 랜덤 에이전트 생성

    num_episodes = 1000  # 에피소드 수
    for episode in range(num_episodes):
        state = env.reset()  # 시작 상태로 초기화
        agent.reset()  # 에이전트 경험 초기화

        while True:
            action = agent.get_action(state)  # 행동 선택
            next_state, reward, done = env.step(action)  # 환경에서 행동 수행
            agent.add(state, action, reward)  # 경험 저장
            if done:  # 에피소드 종료 조건
                agent.eval()  # 에피소드가 끝난 후 가치 함수 업데이트
                break
            state = next_state  # 상태 업데이트

    env.render_v(agent.V)  # 가치 함수 시각화