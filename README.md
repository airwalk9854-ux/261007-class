# 개념 기반 탐구 질문 학습 앱

교과 설계 정보와 참고 자료를 바탕으로 학생의 탐구 질문을 분석하고 피드백하는 Streamlit 앱입니다. 질문과 참고 자료 분석에는 OpenAI API의 `gpt-4o-mini` 모델을 사용합니다.

## API 키 설정

앱에서 AI 분석을 사용하려면 `OPENAI_API_KEY`를 설정해야 합니다.

- **Streamlit Community Cloud:** 앱의 **Settings → Secrets**에 아래 항목을 등록합니다.
  ```toml
  OPENAI_API_KEY = "실제_API_키"
  ```
- **로컬 실행:** `.streamlit/secrets.toml` 파일에 같은 내용을 저장합니다. 이 파일은 Git에 커밋하지 마세요.

학생 질문 분석 시 질문, 교과 설계 정보, 저장된 참고 자료 요약이 OpenAI로 전송됩니다. 학생 이름과 학번은 분석 요청에 포함하지 않습니다. 교사가 PDF 분석을 요청하면 추출된 문서 텍스트와 교과 설계 정보가 전송됩니다. 개인정보나 민감한 내용이 포함된 자료는 업로드하지 마세요.

## 로컬에서 실행

1. 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
2. 앱을 실행합니다.
   ```bash
   streamlit run streamlit_app.py
   ```
