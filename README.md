<div align="center">

# CBNU Industrial AI Coursework Portfolio

충북대학교 산업인공지능학과 석사과정 수업 결과물을 정리한 연구·실습 포트폴리오

<p>
	<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/OpenCV-Practice-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
	<img src="https://img.shields.io/badge/Reinforcement%20Learning-Labs-111827?style=for-the-badge" alt="Reinforcement Learning" />
</p>

</div>

이 저장소는 단순한 코드 모음이 아니라, 각 수업에서 어떤 문제를 다루었고 어떤 방식으로 실험했는지를 빠르게 파악할 수 있도록 구성했습니다. 학습 과정에서 시도한 알고리즘, 구현 예제, 결과 문서를 함께 모아두었습니다.

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

- `01_Image Input-Output & GUI`: OpenCV 기초, 이미지 입출력, 도형 그리기, 인터랙티브 예제
- `02_Histogram & Filtering`: 히스토그램 분석 및 필터링 실습
- `03_Frequency-based Image Filtering`: Sobel, sharpen mask, Gabor filter, DFT, thresholding, binary operation, morphology
- `04_Boundary Extraction`: point set distance, contour, connected component
- `05_Image Segmentation`: Canny edge detection, GrabCut, Hough transform, K-means, watershed
- `06_Feature Detection`: 특징점, descriptor(SURF/BRIEF/ORB), RANSAC, matching 관련 실습
- `08_Optical flow & Panorama stitching`: affine/perspective warp, optical flow, panorama stitching
- `09_Geometric Camera Models`: pinhole/fisheye 카메라 모델, 캘리브레이션, 렌즈 왜곡 보정
- `10_Epipolar Geometry`: fundamental/essential matrix, PnP, stereo rectification, 3D triangulation
- `11_Object Detector`: ArUco marker, 얼굴 검출, OCR, HOG 기반 보행자 검출
- `FinalProject`: 학기말 프로젝트 (Stable Virtual Camera 기반)

### 02_RainforceLearning

강화학습의 대표적인 기본 알고리즘을 실습한 폴더입니다.

- `01_Bandit`: epsilon-greedy bandit 실험
- `02_DynamicProgramming`: policy evaluation, policy iteration, value iteration
- `03_MonteCarloMethod`: Monte Carlo 기반 정책 평가와 제어

## Learning Focus

이 저장소는 "석사과정에서 무엇을 학습했는가"를 중심으로 구성되어 있습니다.

- 영상처리: 공간/주파수 도메인 처리, 세그멘테이션, 특징 추출과 매칭, 기하 변환
- 강화학습: Bandit부터 DP, MC, TD까지의 핵심 알고리즘 흐름
- 실습 기록: 수업 단원별 코드, 퀴즈, 결과물 문서 아카이빙

## Academic Archive Note

폴더 구조는 강의 진도와 실습 주제를 따라 정리되어 있으며, 각 파일은 특정 개념을 학습하고 검증하기 위한 수업 결과물입니다. 따라서 본 저장소는 서비스 개발용 프로젝트라기보다, 대학원 과정의 학습 궤적을 보존한 기술 아카이브에 가깝습니다.

## Intended Use

충북대학교 산업인공지능학과 석사과정 동안의 학습 내용 정리 및 포트폴리오 아카이빙 목적의 개인 저장소입니다.

## License

개인 학습 및 수업 결과물 보관 목적의 저장소입니다. 외부 공유 또는 재사용 시에는 포함 자료의 출처와 사용 범위를 개별적으로 확인해 주세요.
