"""
bandit_quiz.py 스크립트 설명 PDF 생성기
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 한글 폰트 등록 (Windows의 맑은 고딕 사용)
try:
    malgun_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(malgun_path):
        pdfmetrics.registerFont(TTFont('Malgun', malgun_path))
        korean_font = 'Malgun'
    else:
        # 폰트가 없으면 기본 폰트 사용
        korean_font = 'Helvetica'
        print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
except Exception as e:
    korean_font = 'Helvetica'
    print(f"폰트 등록 오류: {e}")

# PDF 파일 생성
pdf_file = "bandit_quiz_설명서.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                       rightMargin=72, leftMargin=72,
                       topMargin=72, bottomMargin=18)

# 스토리 (문서 내용) 리스트
story = []

# 스타일 정의
styles = getSampleStyleSheet()

# 한글 제목 스타일
title_style = ParagraphStyle(
    'KoreanTitle',
    parent=styles['Title'],
    fontName=korean_font,
    fontSize=24,
    spaceAfter=30,
)

# 한글 제목2 스타일
heading1_style = ParagraphStyle(
    'KoreanHeading1',
    parent=styles['Heading1'],
    fontName=korean_font,
    fontSize=18,
    spaceAfter=12,
    textColor=colors.HexColor('#1f4788'),
)

# 한글 제목3 스타일
heading2_style = ParagraphStyle(
    'KoreanHeading2',
    parent=styles['Heading2'],
    fontName=korean_font,
    fontSize=14,
    spaceAfter=10,
    textColor=colors.HexColor('#2e5ca6'),
)

# 한글 본문 스타일
body_style = ParagraphStyle(
    'KoreanBody',
    parent=styles['BodyText'],
    fontName=korean_font,
    fontSize=11,
    spaceAfter=12,
    leading=16,
)

# 코드 스타일
code_style = ParagraphStyle(
    'Code',
    parent=styles['Code'],
    fontName='Courier',
    fontSize=9,
    leftIndent=20,
    spaceAfter=12,
    textColor=colors.HexColor('#333333'),
    backColor=colors.HexColor('#f5f5f5'),
)

# 제목
story.append(Paragraph("Epsilon 값 비교 실험", title_style))
story.append(Paragraph("bandit_quiz.py 스크립트 설명서", heading2_style))
story.append(Spacer(1, 0.3*inch))

# 1. 개요
story.append(Paragraph("1. 개요", heading1_style))
story.append(Paragraph(
    "본 스크립트는 Multi-Armed Bandit 문제에서 epsilon-greedy 전략의 epsilon 값이 "
    "학습 성능에 미치는 영향을 비교 분석하는 실험입니다. 세 가지 서로 다른 epsilon 값"
    "(0.1, 0.3, 0.01)을 사용하여 각각의 평균 보상률을 측정하고 시각화합니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 2. 실험 목적
story.append(Paragraph("2. 실험 목적", heading1_style))
story.append(Paragraph(
    "이 실험의 주요 목적은 <b>탐험-활용 균형(Exploration-Exploitation Trade-off)</b>에서 "
    "탐험 비율이 학습 성능에 어떤 영향을 미치는지 이해하는 것입니다.",
    body_style
))
story.append(Paragraph(
    "• <b>높은 epsilon (0.3)</b>: 탐험을 많이 하여 더 많은 정보를 수집하지만, "
    "최적 행동을 덜 활용",
    body_style
))
story.append(Paragraph(
    "• <b>중간 epsilon (0.1)</b>: 탐험과 활용의 균형을 맞춤 (일반적으로 권장)",
    body_style
))
story.append(Paragraph(
    "• <b>낮은 epsilon (0.01)</b>: 주로 활용에 집중하여 안정적이지만, "
    "초기 정보가 부정확할 경우 문제 발생 가능",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 3. 코드 구조
story.append(Paragraph("3. 코드 구조 및 동작 원리", heading1_style))

story.append(Paragraph("3.1 run_bandit() 함수", heading2_style))
code_function = """def run_bandit(runs, steps, epsilon):
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
    return np.average(all_rates, axis=0)"""
story.append(Preformatted(code_function, code_style))

story.append(Paragraph(
    "<b>함수 파라미터:</b>",
    body_style
))
story.append(Paragraph(
    "• <b>runs</b>: 독립적인 실험 반복 횟수 (200회)<br/>"
    "• <b>steps</b>: 각 실험당 스텝 수 (1000회)<br/>"
    "• <b>epsilon</b>: 탐험 확률 (0.1, 0.3, 0.01)",
    body_style
))

story.append(Paragraph(
    "<b>반환값:</b> 모든 실험의 평균 보상률 (1000개의 값)",
    body_style
))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("3.2 시뮬레이션 과정", heading2_style))
story.append(Paragraph(
    "<b>Step 1: 초기화</b><br/>"
    "• all_rates 배열(200 x 1000) 생성: 모든 실험의 보상률 저장",
    body_style
))
story.append(Paragraph(
    "<b>Step 2: 실험 반복 (200회)</b><br/>"
    "• 매 실험마다 새로운 Bandit 환경과 Agent 생성<br/>"
    "• 각 실험은 독립적이며 통계적 신뢰도 향상에 기여",
    body_style
))
story.append(Paragraph(
    "<b>Step 3: 각 실험 내 학습 (1000 스텝)</b><br/>"
    "• Agent가 epsilon-greedy 전략으로 행동 선택<br/>"
    "• Bandit으로부터 보상 획득<br/>"
    "• Agent의 Q값 업데이트<br/>"
    "• 누적 평균 보상률 계산",
    body_style
))
story.append(Paragraph(
    "<b>Step 4: 평균 계산</b><br/>"
    "• 200개 실험 결과를 평균하여 최종 학습 곡선 생성",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 4. 세 가지 실험 설정
story.append(Paragraph("4. 세 가지 실험 설정", heading1_style))

code_experiments = """avg_rates_1 = run_bandit(200, 1000, 0.1)
avg_rates_2 = run_bandit(200, 1000, 0.3)
avg_rates_3 = run_bandit(200, 1000, 0.01)"""
story.append(Preformatted(code_experiments, code_style))

# 테이블로 실험 조건 정리
exp_data = [
    ['실험', 'Epsilon', '탐험 비율', '활용 비율', '특징'],
    ['실험 1', '0.1', '10%', '90%', '균형있는 탐험-활용'],
    ['실험 2', '0.3', '30%', '70%', '적극적인 탐험'],
    ['실험 3', '0.01', '1%', '99%', '거의 활용만 수행'],
]

exp_table = Table(exp_data, colWidths=[0.8*inch, 0.9*inch, 1.1*inch, 1.1*inch, 2.2*inch])
exp_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), korean_font),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('FONTNAME', (0, 1), (-1, -1), korean_font),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(exp_table)
story.append(Spacer(1, 0.2*inch))

# 5. 결과 시각화
story.append(Paragraph("5. 결과 시각화", heading1_style))

code_plot = """plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates_1)
plt.plot(avg_rates_2)
plt.plot(avg_rates_3)
plt.legend(["0.1", "0.3", "0.01"])
plt.show()"""
story.append(Preformatted(code_plot, code_style))

story.append(Paragraph(
    "세 개의 학습 곡선이 하나의 그래프에 표시되어 epsilon 값에 따른 성능 차이를 "
    "직관적으로 비교할 수 있습니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 6. 예상 결과 분석
story.append(Paragraph("6. 예상 결과 분석", heading1_style))

story.append(Paragraph("6.1 Epsilon = 0.1 (중간 탐험)", heading2_style))
story.append(Paragraph(
    "• <b>초기 단계</b>: 10%의 탐험으로 다양한 팔을 시도하며 빠르게 학습<br/>"
    "• <b>중기 단계</b>: 좋은 팔을 발견하고 주로 활용하며 보상률 상승<br/>"
    "• <b>후기 단계</b>: 안정적으로 높은 보상률 유지<br/>"
    "• <b>종합</b>: 가장 균형잡힌 성능을 보일 것으로 예상",
    body_style
))

story.append(Paragraph("6.2 Epsilon = 0.3 (높은 탐험)", heading2_style))
story.append(Paragraph(
    "• <b>초기 단계</b>: 많은 탐험으로 정보를 빠르게 수집<br/>"
    "• <b>중기 단계</b>: 최적 팔을 찾았지만 여전히 30% 탐험 수행<br/>"
    "• <b>후기 단계</b>: 불필요한 탐험으로 인해 보상률이 다소 낮을 수 있음<br/>"
    "• <b>종합</b>: 초기에는 좋지만 장기적으로는 낭비가 발생",
    body_style
))

story.append(Paragraph("6.3 Epsilon = 0.01 (낮은 탐험)", heading2_style))
story.append(Paragraph(
    "• <b>초기 단계</b>: 탐험이 적어 초기 학습 속도가 느릴 수 있음<br/>"
    "• <b>중기 단계</b>: 우연히 좋은 팔을 발견하면 빠르게 수렴<br/>"
    "• <b>후기 단계</b>: 초기 선택이 좋았다면 높은 보상률 유지<br/>"
    "• <b>종합</b>: 운에 따라 성능이 크게 달라질 수 있음 (High Variance)",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 7. 핵심 개념 정리
story.append(Paragraph("7. 핵심 개념 정리", heading1_style))

story.append(Paragraph("7.1 탐험-활용 딜레마 (Exploration-Exploitation Dilemma)", heading2_style))
story.append(Paragraph(
    "강화학습의 핵심 문제 중 하나로, 두 가지 목표 사이의 균형을 맞추어야 합니다:",
    body_style
))
story.append(Paragraph(
    "• <b>탐험(Exploration)</b>: 알려지지 않은 행동을 시도하여 더 나은 선택지를 찾음<br/>"
    "• <b>활용(Exploitation)</b>: 현재까지의 정보를 바탕으로 최선의 행동 선택",
    body_style
))
story.append(Paragraph(
    "Epsilon 값은 이 균형을 조절하는 핵심 하이퍼파라미터입니다. "
    "문제의 특성과 환경에 따라 최적의 epsilon 값이 달라질 수 있습니다.",
    body_style
))

story.append(Paragraph("7.2 통계적 신뢰도", heading2_style))
story.append(Paragraph(
    "200번의 독립적인 실험을 평균한 이유:",
    body_style
))
story.append(Paragraph(
    "• 무작위성으로 인한 변동성 감소<br/>"
    "• 알고리즘의 평균적인 성능 파악<br/>"
    "• 재현 가능한 실험 결과 제공<br/>"
    "• 통계적으로 유의미한 비교 가능",
    body_style
))

story.append(Paragraph("7.3 평균 보상률 (Average Reward Rate)", heading2_style))
code_rate = """rates.append(total_reward / (step + 1))"""
story.append(Preformatted(code_rate, code_style))
story.append(Paragraph(
    "누적 보상을 현재까지의 스텝 수로 나눈 값으로, 시간에 따른 학습 진행 상황을 "
    "보여주는 지표입니다. 이 값이 높고 안정적일수록 좋은 학습 성능을 나타냅니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 8. 학습 포인트
story.append(Paragraph("8. 학습 포인트", heading1_style))
story.append(Paragraph(
    "• <b>하이퍼파라미터의 중요성</b>: Epsilon 값 하나만으로도 성능이 크게 달라짐",
    body_style
))
story.append(Paragraph(
    "• <b>실험적 접근</b>: 다양한 설정을 비교하여 최적의 값을 찾는 방법",
    body_style
))
story.append(Paragraph(
    "• <b>시각화의 중요성</b>: 그래프를 통해 직관적으로 성능 차이 파악",
    body_style
))
story.append(Paragraph(
    "• <b>환경 의존성</b>: 같은 epsilon이라도 환경에 따라 성능이 다를 수 있음",
    body_style
))
story.append(Paragraph(
    "• <b>Adaptive Epsilon</b>: 시간에 따라 epsilon을 감소시키는 전략도 가능 (초기 탐험, 후기 활용)",
    body_style
))
story.append(Spacer(1, 0.3*inch))

# 9. 확장 아이디어
story.append(Paragraph("9. 확장 아이디어", heading1_style))
story.append(Paragraph(
    "• 더 많은 epsilon 값 비교 (예: 0.05, 0.2, 0.5)<br/>"
    "• Decaying epsilon: ε = ε₀ × (1 - t/T) 형태로 점진적 감소<br/>"
    "• 다른 탐험 전략 비교: UCB, Thompson Sampling<br/>"
    "• 비정상 환경(Non-stationary)에서의 성능 비교<br/>"
    "• 팔의 개수를 변경하여 실험 (5개, 20개 등)",
    body_style
))
story.append(Spacer(1, 0.3*inch))

# 10. 실행 방법
story.append(Paragraph("10. 실행 방법", heading1_style))
code_run = """python bandit_quiz.py"""
story.append(Preformatted(code_run, code_style))
story.append(Paragraph(
    "스크립트를 실행하면 세 가지 epsilon 값에 대한 실험이 순차적으로 수행되며 "
    "(총 200×3=600번의 독립적인 시뮬레이션), 최종적으로 세 개의 학습 곡선이 "
    "비교되는 그래프가 표시됩니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph(
    "<b>주의사항:</b> 실행 시간은 컴퓨터 성능에 따라 다르지만, "
    "일반적으로 수 초에서 수십 초 정도 소요됩니다.",
    body_style
))

# PDF 생성
doc.build(story)
print(f"PDF 파일이 생성되었습니다: {pdf_file}")
