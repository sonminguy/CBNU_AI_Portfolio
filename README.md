<div align="center">

# CBNU Industrial AI Coursework Portfolio

충북대학교 산업인공지능학과 석사과정 수업 결과물을 정리한 연구·실습 포트폴리오

<p>
	<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/OpenCV-Practice-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
	<img src="https://img.shields.io/badge/Reinforcement%20Learning-Labs-111827?style=for-the-badge" alt="Reinforcement Learning" />
</p>

</div>

이 저장소는 단순한 코드 모음이 아니라, 각 수업에서 어떤 문제를 다루었고 어떤 방식으로 실험했는지를 빠르게 파악할 수 있도록 구성했습니다. 영상처리와 강화학습이라는 두 큰 축을 중심으로, 학습 과정에서 시도한 알고리즘, 구현 예제, 결과 문서를 함께 모아두었습니다.

## Repository Snapshot

| Area | Main Topics | Purpose |
| --- | --- | --- |
| `01_ImageProcessing` | OpenCV, 필터링, 주파수 변환, 윤곽선, 모폴로지, 특징점, 정합, 광류 | 영상처리 수업 실습 및 과제 정리 |
| `02_RainforceLearning` | Bandit, Dynamic Programming, Monte Carlo, Temporal Difference | 강화학습 기초 알고리즘 실습 및 실험 결과 정리 |

## Highlights

- 실습 중심의 파이썬 구현 코드
- 수업별로 분리된 폴더 구조
- 실험 결과와 설명 문서를 함께 보관
- OpenCV, NumPy, Matplotlib 기반의 가벼운 재현 환경

## Folder Guide

### 01_ImageProcessing

영상처리 수업에서 다룬 내용을 단계적으로 정리한 폴더입니다.

- `01_cv1` `02_cv2`: OpenCV 기초, 이미지 입출력, 도형 그리기, 인터랙티브 예제
- `03_cv3`: Sobel, sharpen mask, Gabor filter, DFT, thresholding, binary operation, morphology
- `04_cv4`: point set distance, contour, connected component
- `05_cv5`: Canny edge detection, GrabCut, Hough transform, K-means, watershed
- `06_cv6`: 특징점, descriptor, RANSAC, matching 관련 실습
- `07_cv7`: affine transform, optical flow, panorama stitching, remap
- `99_Document`: 참고 문서 및 부가 자료

### 02_RainforceLearning

강화학습의 대표적인 기본 알고리즘을 실습한 폴더입니다.

- `01_Bandit`: epsilon-greedy bandit 실험
- `02_DynamicProgramming`: policy evaluation, policy iteration, value iteration
- `03_MonteCarloMethod`: Monte Carlo 기반 정책 평가와 제어
- `04_TemporalDifferenceMethod`: SARSA, Q-learning, TD evaluation
- `99_Document`: 실습 자료 및 참고 문서

## How to Run

대부분의 코드는 Python 환경에서 동작합니다. 예시는 아래와 같습니다.

```bash
python 01_ImageProcessing/01_cv1/Lena.py
python 02_RainforceLearning/01_Bandit/bandit.py
```

실행 전 필요한 경우 다음 패키지를 설치하세요.

```bash
pip install numpy matplotlib opencv-python
```

## Notes

- 일부 스크립트는 데이터 파일 경로를 기준으로 동작하므로, 실행 위치를 각 폴더 기준으로 맞추는 것이 좋습니다.
- `.zip` 파일은 과제 제출 또는 배포용 결과물로 포함되어 있습니다.
- `data` 폴더에는 예제 이미지와 실험 입력 파일이 들어 있습니다.

## Intended Use

이 저장소는 학습 기록과 포트폴리오 정리를 위한 개인 연구/수업용 저장소입니다. 필요하다면 각 폴더에 대해 별도의 실행 방법, 결과 이미지, 핵심 알고리즘 설명을 추가해 GitHub 프로필형 포트폴리오로 확장할 수 있습니다.

## License

개인 학습 및 수업 결과물 보관을 위한 저장소입니다. 외부 공유 시에는 포함된 자료와 코드의 출처 및 사용 범위를 별도로 확인해 주세요.
