import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SETTINGS_PATH = DATA_DIR / "teacher_settings.json"
STUDENT_PATH = DATA_DIR / "student_questions.csv"

st.set_page_config(page_title="개념 기반 탐구 질문 앱", page_icon="🔍")


def ensure_storage():
    DATA_DIR.mkdir(exist_ok=True)
    if not SETTINGS_PATH.exists():
        default_settings = {
            "achievement_standards": "2022 개정 교육과정에서 핵심 성취기준을 입력하세요.",
            "core_concepts": "핵심 개념을 입력하세요.",
            "core_ideas": "핵심 아이디어를 입력하세요.",
            "core_questions": "핵심 질문을 입력하세요.",
            "reference_materials": "",
        }
        SETTINGS_PATH.write_text(json.dumps(default_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    if not STUDENT_PATH.exists():
        empty_df = pd.DataFrame(
            columns=[
                "학번",
                "이름",
                "초기질문",
                "질문유형",
                "점수",
                "피드백",
                "수정질문",
                "수정질문유형",
                "수정질문점수",
                "제출시각",
            ]
        )
        empty_df.to_csv(STUDENT_PATH, index=False, encoding="utf-8")


ensure_storage()


def load_teacher_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "achievement_standards": "",
            "core_concepts": "",
            "core_ideas": "",
            "core_questions": "",
        }


def save_teacher_settings(settings):
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def load_student_records():
    try:
        df = pd.read_csv(STUDENT_PATH, encoding="utf-8")
        if df.empty:
            return pd.DataFrame(columns=[
                "학번",
                "이름",
                "초기질문",
                "질문유형",
                "점수",
                "피드백",
                "수정질문",
                "수정질문유형",
                "수정질문점수",
                "제출시각",
            ])
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "학번",
            "이름",
            "초기질문",
            "질문유형",
            "점수",
            "피드백",
            "수정질문",
            "수정질문유형",
            "수정질문점수",
            "제출시각",
        ])


def save_student_record(row):
    df = load_student_records()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STUDENT_PATH, index=False, encoding="utf-8")


def classify_question(question_text):
    text = question_text.strip()
    lower = text.lower()
    if not text:
        return "사실적 질문", 0, "질문을 입력해주세요."

    argument_keywords = [
        "논쟁", "의견", "판단", "비판", "토론", "어떻게 생각", "옳다", "틀리다",
        "가치", "관점", "중요한 이유", "우선순위", "비교해서 어떤", "문제점"
    ]
    concept_keywords = [
        "왜", "어떻게", "무엇을 의미", "비교", "공통점", "차이점", "원인", "결과",
        "관계", "개념", "원리", "변화", "과정", "의미", "연결"
    ]
    factual_keywords = [
        "누가", "언제", "어디", "어떤", "무엇", "몇", "몇 개", "예를 들어",
        "정의", "사실", "증거", "기록", "상세"
    ]

    if any(keyword in lower for keyword in argument_keywords):
        qtype = "논쟁적 질문"
    elif any(keyword in lower for keyword in concept_keywords):
        qtype = "개념적 질문"
    else:
        qtype = "사실적 질문"

    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    score = 0
    if qtype == "사실적 질문":
        score = 45
        if word_count >= 12:
            score += 6
        if any(keyword in lower for keyword in ["왜", "어떻게", "비교", "관계", "원인", "결과"]):
            score += 6
        if word_count < 8:
            score -= 10
    elif qtype == "개념적 질문":
        score = 68
        if word_count >= 14:
            score += 6
        if any(keyword in lower for keyword in ["비교", "원인", "결과", "관계", "공통점", "차이점", "의미"]):
            score += 6
        if any(keyword in lower for keyword in ["어떻게", "왜"]):
            score += 4
        if word_count < 8:
            score -= 12
    else:
        score = 85
        if word_count >= 15:
            score += 5
        if any(keyword in lower for keyword in ["관점", "비판", "의견", "비교", "토론", "가치", "판단"]):
            score += 5
        if word_count < 8:
            score -= 15

    if score < 1:
        score = 1
    if score > 100:
        score = 100

    if word_count <= 6 and qtype != "논쟁적 질문":
        return qtype, max(1, score - 10), "질문이 너무 짧아서 탐구를 확장할 기회가 있습니다."

    return qtype, int(score), "분석 완료"


def extract_top_sentences(text, limit=3):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned[:limit]


def summarize_pdf_text(text):
    if not text or len(text.strip()) < 10:
        return "추출된 문서 내용이 없습니다."
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    top = [s.strip() for s in sentences if len(s.strip()) > 20][:4]
    return " ".join(top)


def build_feedback(question_text, question_type, score, teacher_settings, pdf_texts=None):
    pdf_texts = pdf_texts or []
    teacher_context = "\n".join(
        [
            teacher_settings.get("achievement_standards", ""),
            teacher_settings.get("core_concepts", ""),
            teacher_settings.get("core_ideas", ""),
            teacher_settings.get("core_questions", ""),
        ]
    )

    pdf_summary = "\n".join([summarize_pdf_text(t) for t in pdf_texts if t])
    fact_background = "\n".join(
        [
            teacher_context,
            pdf_summary,
        ]
    )

    if question_type == "사실적 질문":
        feedback = (
            "이 질문은 기본 사실이나 예시를 확인하는 데 적합합니다. 다만 학생들이 단순히 답을 찾는 수준을 넘어서 "
            "왜, 어떻게, 어떤 조건에서, 어떤 사례와 연결되는지까지 확장하면 탐구 깊이가 높아집니다."
        )
        guidance = [
            "관련된 핵심 개념이나 단원 개념을 한 문장으로 연결해 보세요.",
            "이 질문이 왜 중요한지를 설명하는 문장을 추가해 보세요.",
            "사례를 하나 더 제시해 보고, 비슷한 상황과 다른 상황을 비교해 보세요.",
        ]
    elif question_type == "개념적 질문":
        feedback = (
            "이 질문은 사실을 이해하고 추론하는 흐름으로 발전할 가능성이 큽니다. "
            "학습한 사실이 어떤 원리와 연결되는지, 그리고 다른 상황에 어떻게 적용되는지를 함께 설명하면 더 깊은 탐구가 됩니다."
        )
        guidance = [
            "이 개념이 다른 사례에서도 적용되는지 생각해 보세요.",
            "원인과 결과 사이의 관계를 언급해 보세요.",
            "비슷한 개념과 차이점을 비교하면서 질문의 범위를 넓혀 보세요.",
        ]
    else:
        feedback = (
            "이 질문은 논쟁적이면서 비판적 사고를 유도하는 방향으로 발전하고 있습니다. "
            "다양한 관점을 이해하고 근거를 함께 제시하면 더 정교한 탐구 질문이 됩니다."
        )
        guidance = [
            "서로 다른 관점에서 이 문제를 어떻게 볼 수 있는지 정리해 보세요.",
            "근거가 되는 사실과 사례를 각각 제시해 보세요.",
            "이 질문이 학생들에게 어떤 가치 판단이나 선택을 요구하는지 점검해 보세요.",
        ]

    if score < 60:
        score_note = "질문이 단순 사실 확인에 치우쳐 탐구의 폭이 좁습니다."
    elif score < 80:
        score_note = "질문이 탐구를 시작하기에 적절하지만, 연결과 추론을 더 강화하면 좋습니다."
    else:
        score_note = "질문이 깊이 있는 탐구를 뒷받침하는 수준입니다."

    related_facts = []
    if teacher_context:
        related_facts.append(teacher_context[:180])
    if pdf_summary:
        related_facts.append(pdf_summary[:180])

    fact_text = "\n".join([f"- {item}" for item in related_facts[:2]]) if related_facts else "- 관련 사실을 보강해 주세요."

    detailed_feedback = (
        f"분류: {question_type}\n"
        f"학술적 깊이 점수: {score}/100\n\n"
        f"{feedback}\n\n"
        f"관련 사실 및 배경 지식:\n{fact_text}\n\n"
        f"질문 발전 제안:\n"
        + "\n".join(f"{idx}. {item}" for idx, item in enumerate(guidance, start=1))
        + f"\n\n{score_note}"
    )
    return detailed_feedback


def render_student_login():
    st.title("중학생 개념 기반 탐구 질문 학습 앱")

    st.subheader("학생 로그인")
    st.write("학번과 이름을 입력한 뒤, 탐구 질문을 작성하고 질문을 발전시켜 보세요.")

    with st.form("student_login_form"):
        name = st.text_input("이름")
        student_id = st.text_input("학번")
        submitted = st.form_submit_button("로그인")

    if submitted:
        if not name.strip() or not student_id.strip():
            st.warning("이름과 학번을 모두 입력해 주세요.")
        else:
            st.session_state.student_name = name.strip()
            st.session_state.student_id = student_id.strip()
            st.session_state.page = "student"
            st.rerun()

    st.markdown("---")
    st.subheader("교사용 대시보드 접속")
    st.write("교사는 별도 비밀번호로만 대시보드에 진입할 수 있습니다.")

    with st.form("teacher_access_form"):
        teacher_password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
        go_teacher = st.form_submit_button("입장")

    if go_teacher:
        if teacher_password == "200208":
            st.session_state.page = "teacher"
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")

    with st.sidebar:
        st.markdown("---")
        st.caption("학생용 화면")


def render_teacher_password():
    st.title("교사용 대시보드 접근")
    st.write("비밀번호를 입력해 주세요.")
    password = st.text_input("비밀번호", type="password")
    if st.button("입장"):
        if password == "200208":
            st.session_state.page = "teacher"
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")

    with st.sidebar:
        if st.button("학생 로그인으로 돌아가기"):
            st.session_state.page = "student_login"
            st.rerun()


def render_teacher_dashboard():
    st.title("교사용 대시보드")
    settings = load_teacher_settings()
    student_df = load_student_records()

    with st.sidebar:
        if st.button("로그아웃 및 학생 화면으로 이동"):
            for key in ["student_name", "student_id", "page", "initial_question", "analysis_done"]:
                st.session_state.pop(key, None)
            st.session_state.page = "student_login"
            st.rerun()

    st.subheader("1. 교과 설계 정보")
    with st.form("teacher_settings_form"):
        achievement_standards = st.text_area("2022 개정 교육과정 성취기준", value=settings.get("achievement_standards", ""))
        core_concepts = st.text_area("핵심 개념", value=settings.get("core_concepts", ""))
        core_ideas = st.text_area("핵심 아이디어", value=settings.get("core_ideas", ""))
        core_questions = st.text_area("핵심 질문", value=settings.get("core_questions", ""))
        save_button = st.form_submit_button("설정 저장")

    if save_button:
        settings.update(
            {
                "achievement_standards": achievement_standards,
                "core_concepts": core_concepts,
                "core_ideas": core_ideas,
                "core_questions": core_questions,
                "reference_materials": settings.get("reference_materials", ""),
            }
        )
        save_teacher_settings(settings)
        st.success("교과 설계 정보가 저장되었습니다. 이 정보는 학생 질문 피드백에 반영됩니다.")

    st.subheader("2. PDF 자료 업로드 및 분석")
    pdf_files = st.file_uploader("교과서 파일 또는 참고 문서 PDF 업로드", type=["pdf"], accept_multiple_files=True)
    pdf_summaries = []
    if pdf_files:
        for uploaded_file in pdf_files:
            bytes_data = uploaded_file.read()
            from io import BytesIO
            reader = PdfReader(BytesIO(bytes_data))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            pdf_text = "\n".join(pages)
            pdf_summaries.append((uploaded_file.name, summarize_pdf_text(pdf_text)))

        reference_materials = "\n\n".join([f"{name}: {summary}" for name, summary in pdf_summaries])
        settings["reference_materials"] = reference_materials
        save_teacher_settings(settings)

        for name, summary in pdf_summaries:
            with st.expander(f"{name}"):
                st.write(summary)

    st.subheader("3. 학생 질문 현황")
    if student_df.empty:
        st.info("아직 제출된 학생 질문이 없습니다.")
    else:
        st.dataframe(student_df, width=1200, height=400)

        counts = student_df["질문유형"].value_counts().rename_axis("질문유형").reset_index(name="개수")
        st.bar_chart(counts.set_index("질문유형"))

        st.subheader("학생별 피드백 확인")
        for _, row in student_df.iterrows():
            with st.expander(f"{row['이름']}({row['학번']}) - {row['질문유형']}"):
                st.write("초기 질문:")
                st.write(row["초기질문"])
                st.write("피드백:")
                st.write(row["피드백"])
                if str(row.get("수정질문", "")).strip():
                    st.write("수정 질문:")
                    st.write(row["수정질문"])


def render_student_app():
    st.title("탐구 질문 작성 및 발전")
    student_name = st.session_state.get("student_name", "학생")
    student_id = st.session_state.get("student_id", "")
    st.caption(f"로그인 학생: {student_name} ({student_id})")

    with st.sidebar:
        if st.button("로그아웃"):
            for key in ["student_name", "student_id", "page", "initial_question", "analysis_done"]:
                st.session_state.pop(key, None)
            st.session_state.page = "student_login"
            st.rerun()

    teacher_settings = load_teacher_settings()
    pdf_context = teacher_settings.get("reference_materials", "")

    st.subheader("1단계. 탐구 질문 작성")
    initial_question = st.text_area(
        "처음으로 생각한 탐구 질문을 입력해 주세요.",
        value=st.session_state.get("initial_question", ""),
        height=140,
    )
    if st.button("질문 분석하기"):
        st.session_state.initial_question = initial_question
        st.session_state.analysis_done = True
        st.session_state.question_type, st.session_state.question_score, _ = classify_question(initial_question)
        st.session_state.feedback = build_feedback(
            initial_question,
            st.session_state.question_type,
            st.session_state.question_score,
            teacher_settings,
            [pdf_context] if pdf_context else [],
        )

    if st.session_state.get("analysis_done"):
        st.success("질문 분석이 완료되었습니다. 아래 가이드를 확인해 보세요.")

    st.subheader("2단계. 질문 유형 가이드")
    type_cols = st.columns(3)
    with type_cols[0]:
        st.markdown("### 사실적 질문")
        st.write("단원의 개념이나 예를 물어보는 질문입니다. 기본 지식을 확인하는 데 유용합니다.")
    with type_cols[1]:
        st.markdown("### 개념적 질문")
        st.write("사실을 연결하고 추론하며, 전이 가능한 개념적 이해를 탐색하는 질문입니다.")
    with type_cols[2]:
        st.markdown("### 논쟁적 질문")
        st.write("답이 정해져 있지 않고 다양한 관점이 필요한 질문입니다. 비판적 사고를 자극합니다.")

    if st.session_state.get("analysis_done"):
        st.info(f"분석 결과: {st.session_state.question_type} / 학술적 깊이: {st.session_state.question_score}/100")

    st.subheader("3단계. AI 피드백 및 질문 발전 가이드")
    if st.session_state.get("analysis_done"):
        progress_value = min(max(st.session_state.question_score, 1), 100)
        st.progress(progress_value / 100)
        st.metric("학술적 깊이 점수", f"{progress_value}/100")
        st.write(st.session_state.feedback)
    else:
        st.info("질문을 분석하면 AI 피드백이 표시됩니다.")

    st.subheader("4단계. 수정 질문 입력 및 제출")
    revised_question = st.text_area("피드백을 반영해 수정한 질문을 다시 입력해 주세요.", height=140)
    if st.button("수정 질문 제출"):
        if not revised_question.strip():
            st.warning("수정한 질문을 입력해 주세요.")
        else:
            revised_type, revised_score, _ = classify_question(revised_question)
            revised_feedback = build_feedback(
                revised_question,
                revised_type,
                revised_score,
                teacher_settings,
                [pdf_context] if pdf_context else [],
            )
            st.session_state.revised_type = revised_type
            st.session_state.revised_score = revised_score
            st.session_state.revised_feedback = revised_feedback

            row = {
                "학번": student_id,
                "이름": student_name,
                "초기질문": initial_question,
                "질문유형": st.session_state.get("question_type", ""),
                "점수": st.session_state.get("question_score", 0),
                "피드백": st.session_state.get("feedback", ""),
                "수정질문": revised_question,
                "수정질문유형": revised_type,
                "수정질문점수": revised_score,
                "제출시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_student_record(row)
            st.success("질문이 제출되었습니다. 교사용 대시보드에서 확인할 수 있습니다.")
            st.write("수정 질문 분석 결과")
            st.info(f"유형: {revised_type} / 점수: {revised_score}/100")
            st.write(revised_feedback)


page = st.session_state.get("page", "student_login")

if page == "student_login":
    render_student_login()
elif page == "teacher_password":
    render_teacher_password()
elif page == "teacher":
    render_teacher_dashboard()
else:
    render_student_app()
