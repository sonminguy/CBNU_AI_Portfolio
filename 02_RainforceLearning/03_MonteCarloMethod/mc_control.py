import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

def greedy_probs(Q, state, epsilon = 0, action_size = 4):
    qs = [Q[(state, action)] for action in range(action_size)]  # 현재 상태에서 각 행동의 Q값 가져오기
    max_action = np.argmax(qs)  # Q값이 가장 높은 행동 선택
    best_prob = epsilon / action_size  # 탐험 행동 확률 계산
    action_probs = {actions: best_prob for actions in range(action_size)}  # 모든 행동에 탐험 행동 확률 할당
    action_probs[max_action] += (1.0 - epsilon)  # 최적 행동
    return action_probs  # 행동 확률 반환

class MCAgent:
    def __init__(self):
        self.gamma = 0.9  # 할인율
        self.epsilon = 0.1  # 탐험 확률
        self.alpha = 0.1  # 학습률
        self.action_size = 4  # 행동 공간 크기

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 행동 확률 (균등 분포)
        self.pi = defaultdict(lambda: random_actions)  # 행동 정책 (초기값 균등 분포)
        self.Q = defaultdict(lambda: 0)  # 상태-행동 가치 함수 (초기값 0)
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

    def update(self):
        G = 0  # 누적 보상 초기화
        for state, action, reward in reversed(self.memory):  # 에피소드 경험을 역순으로 처리
            G = reward + self.gamma * G  # 누적 보상 계산
            key = (state, action)  # 상태-행동 쌍
            self.Q[key] += self.alpha * (G - self.Q[key])  # 상태-행동 가치 함수 업데이트
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)  # 행동 정책 업데이트


if __name__ == "__main__":
    env = GridWorld()  # 그리드 월드 환경 생성
    agent = MCAgent()  # 랜덤 에이전트 생성

    episodes = 1000  # 에피소드 수
    for episode in range(episodes):
        state = env.reset()  # 시작 상태로 초기화
        agent.reset()  # 에이전트 경험 초기화

        while True:
            action = agent.get_action(state)  # 행동 선택
            next_state, reward, done = env.step(action)  # 환경에서 행동 수행
            agent.add(state, action, reward)  # 경험 저장
            if done:  # 에피소드 종료 조건
                agent.update()  # 에피소드가 끝난 후 가치 함수 업데이트
                break
            state = next_state  # 상태 업데이트

    env.render_q(agent.Q)  # 가치 함수 시각화