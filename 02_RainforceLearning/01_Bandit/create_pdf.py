"""
bandit_avg.py 스크립트 설명 PDF 생성기
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
pdf_file = "bandit_avg_설명서.pdf"
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
story.append(Paragraph("Multi-Armed Bandit 시뮬레이션", title_style))
story.append(Paragraph("bandit_avg.py 스크립트 설명서", heading2_style))
story.append(Spacer(1, 0.3*inch))

# 1. 개요
story.append(Paragraph("1. 개요", heading1_style))
story.append(Paragraph(
    "본 스크립트는 강화학습의 기초 문제인 Multi-Armed Bandit 문제를 epsilon-greedy 전략을 "
    "사용하여 해결하는 시뮬레이션입니다. 여러 번의 독립적인 실험을 수행하고 평균 보상률을 "
    "분석하여 시각화합니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 2. 주요 개념
story.append(Paragraph("2. 주요 개념", heading1_style))

story.append(Paragraph("2.1 Multi-Armed Bandit 문제", heading2_style))
story.append(Paragraph(
    "Multi-Armed Bandit은 여러 개의 슬롯 머신(팔) 중에서 어떤 것을 선택해야 최대 보상을 "
    "얻을 수 있는지 학습하는 문제입니다. 각 팔은 서로 다른 보상 확률을 가지고 있으며, "
    "에이전트는 시행착오를 통해 최적의 팔을 찾아야 합니다.",
    body_style
))

story.append(Paragraph("2.2 Epsilon-Greedy 전략", heading2_style))
story.append(Paragraph(
    "Epsilon-greedy는 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추는 전략입니다:",
    body_style
))
story.append(Paragraph("• <b>활용(Exploitation)</b>: 현재까지 가장 좋은 것으로 추정되는 팔을 선택 (확률 1-ε)", body_style))
story.append(Paragraph("• <b>탐험(Exploration)</b>: 무작위로 팔을 선택하여 새로운 정보 수집 (확률 ε)", body_style))
story.append(Paragraph(
    "본 스크립트에서는 ε=0.1로 설정하여 90%는 최선의 선택을, 10%는 탐험을 수행합니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 3. 코드 구조
story.append(Paragraph("3. 코드 구조 및 동작 원리", heading1_style))

story.append(Paragraph("3.1 시뮬레이션 파라미터", heading2_style))
code_params = """runs = 200      # 독립적인 실험 반복 횟수
steps = 1000    # 각 실험당 스텝 수
epsilon = 0.1   # 탐험 확률 (10%)"""
story.append(Preformatted(code_params, code_style))
story.append(Paragraph(
    "• <b>runs</b>: 통계적 신뢰도를 높이기 위해 200번의 독립적인 실험 수행<br/>"
    "• <b>steps</b>: 각 실험마다 1000번의 행동 선택 수행<br/>"
    "• <b>epsilon</b>: 10% 확률로 무작위 탐험 수행",
    body_style
))

story.append(Paragraph("3.2 주요 클래스", heading2_style))
story.append(Paragraph(
    "<b>Bandit 클래스</b>: 10개의 팔을 가진 슬롯 머신 환경을 시뮬레이션합니다. "
    "각 팔은 무작위로 생성된 보상 확률을 가지고 있으며, play() 메서드를 통해 "
    "선택된 팔에 대한 보상(0 또는 1)을 반환합니다.",
    body_style
))
story.append(Paragraph(
    "<b>Agent 클래스</b>: epsilon-greedy 전략을 사용하여 행동을 선택하는 에이전트입니다. "
    "각 팔의 가치(Q값)를 추정하고, get_action()으로 행동을 선택하며, "
    "update()로 경험을 통해 학습합니다.",
    body_style
))

story.append(Paragraph("3.3 시뮬레이션 루프", heading2_style))
story.append(Paragraph(
    "이중 루프 구조로 되어 있습니다:",
    body_style
))
story.append(Paragraph(
    "1. <b>외부 루프 (runs)</b>: 200번의 독립적인 실험 반복<br/>"
    "   - 매 실험마다 새로운 Bandit과 Agent 생성<br/>"
    "   - 누적 보상과 보상률 초기화",
    body_style
))
story.append(Paragraph(
    "2. <b>내부 루프 (steps)</b>: 각 실험당 1000번의 스텝 수행<br/>"
    "   - Agent가 행동 선택 (epsilon-greedy)<br/>"
    "   - Bandit으로부터 보상 획득<br/>"
    "   - Agent의 가치 추정 업데이트<br/>"
    "   - 평균 보상률 계산: total_reward / (step + 1)",
    body_style
))

story.append(Paragraph("3.4 가치 함수 업데이트", heading2_style))
code_update = """self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]"""
story.append(Preformatted(code_update, code_style))
story.append(Paragraph(
    "점진적 평균(Incremental Mean) 방식으로 Q값을 업데이트합니다. "
    "이는 메모리 효율적이며 실시간 학습에 적합합니다. "
    "새로운 보상이 들어올 때마다 기존 추정값과의 차이를 선택 횟수로 나누어 반영합니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 4. 결과 분석
story.append(Paragraph("4. 결과 분석", heading1_style))
story.append(Paragraph(
    "<b>평균 보상률 그래프</b>: 200번의 실험에 대한 평균을 계산하여 "
    "각 스텝별 평균 보상률의 변화를 시각화합니다. "
    "그래프는 학습이 진행됨에 따라 보상률이 점차 증가하고 안정화되는 모습을 보여줍니다.",
    body_style
))
story.append(Paragraph(
    "초기에는 탐험을 통해 좋은 팔을 찾아가는 과정에서 변동성이 크지만, "
    "시간이 지남에 따라 최적의 팔에 수렴하면서 보상률이 안정화됩니다. "
    "이는 epsilon-greedy 전략이 효과적으로 작동하고 있음을 보여줍니다.",
    body_style
))
story.append(Spacer(1, 0.2*inch))

# 5. 핵심 알고리즘
story.append(Paragraph("5. 핵심 알고리즘 정리", heading1_style))

# 테이블 생성
algo_data = [
    ['단계', '설명', '수식'],
    ['1. 초기화', 'Q값과 선택 횟수 초기화', 'Q(a) = 0, N(a) = 0'],
    ['2. 행동 선택', 'ε 확률로 무작위, 1-ε 확률로 최대 Q값', 'a = argmax Q(a)'],
    ['3. 보상 획득', '선택한 팔에서 보상 받기', 'r ∈ {0, 1}'],
    ['4. 가치 업데이트', '점진적 평균으로 Q값 갱신', 'Q(a) ← Q(a) + (r - Q(a))/N(a)'],
    ['5. 반복', '충분히 학습될 때까지 2-4 반복', '-'],
]

algo_table = Table(algo_data, colWidths=[1.2*inch, 2.5*inch, 2*inch])
algo_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), korean_font),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTNAME', (0, 1), (-1, -1), korean_font),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(algo_table)
story.append(Spacer(1, 0.2*inch))

# 6. 학습 포인트
story.append(Paragraph("6. 학습 포인트", heading1_style))
story.append(Paragraph(
    "• <b>탐험-활용 딜레마</b>: 새로운 정보를 얻기 위한 탐험과 현재 최선의 선택을 하는 활용 사이의 균형",
    body_style
))
story.append(Paragraph(
    "• <b>점진적 학습</b>: 모든 데이터를 저장하지 않고도 평균을 계산하는 효율적인 방법",
    body_style
))
story.append(Paragraph(
    "• <b>통계적 검증</b>: 여러 번의 독립적인 실험을 통해 결과의 신뢰도 향상",
    body_style
))
story.append(Paragraph(
    "• <b>비정상성(Non-stationarity)</b>: 환경이 변하지 않는 정상 문제에 대한 기본적인 접근",
    body_style
))
story.append(Spacer(1, 0.3*inch))

# 7. 실행 방법
story.append(Paragraph("7. 실행 방법", heading1_style))
code_run = """python bandit_avg.py"""
story.append(Preformatted(code_run, code_style))
story.append(Paragraph(
    "스크립트를 실행하면 200번의 실험이 자동으로 수행되고, "
    "평균 보상률을 나타내는 그래프가 표시됩니다.",
    body_style
))

# PDF 생성
doc.build(story)
print(f"PDF 파일이 생성되었습니다: {pdf_file}")
