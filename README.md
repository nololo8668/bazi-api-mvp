# 만세력 계산기 & AI 사주 서비스

전통 명리학의 만세력(四柱, BaZi)을 계산하고 AI 기반 해석을 제공하는 웹 서비스입니다.

## 🚀 주요 기능

- **만세력 계산**: 생년월일시 기반 네 기둥(연/월/일/시) 계산
- **지장간 분석**: 각 지지에 숨겨진 천간 분석
- **오행 분포**: 목/화/토/금/수 오행 집계 및 분석
- **십성 해석**: 비견/겁재/식신 등 십성 분석
- **대운 계산**: 10년 단위 운세 변화 분석
- **AI 해석**: OpenAI GPT를 활용한 한국어 사주 해석
- **정밀 계산**: 절기 경계와 진태양시 보정 지원

## 🏗️ 프로젝트 구조

```
만세력모듈/
├── api_server.py          # FastAPI 백엔드 서버
├── bazi_step6.py          # 핵심 사주 계산 로직
├── public/
│   └── index.html         # 프론트엔드 MVP
├── tests_golden.py        # 골든 테스트
├── requirements.txt        # Python 의존성
└── .gitignore.txt         # Git 제외 파일
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트에 `.env` 파일 생성:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 3. 서버 실행
```bash
uvicorn api_server:app --reload
```

### 4. 접속
- 웹 UI: http://127.0.0.1:8000
- API 문서: http://127.0.0.1:8000/docs
- 헬스체크: http://127.0.0.1:8000/health

## 📡 API 엔드포인트

### POST /api/bazi/calc
문자열 입력으로 사주 계산:
```json
{
  "birth_local": "1982-05-25 20:00:00",
  "tz": "Asia/Seoul",
  "sex": "M",
  "include_text_report": true
}
```

### POST /api/bazi/analyze
폼 파라미터로 사주 계산:
```json
{
  "year": 1982,
  "month": 5,
  "day": 25,
  "hour": 20,
  "sex": "M",
  "include_ai": true
}
```

## 🔧 주요 옵션

- **calendar**: "solar" (양력) / "lunar" (음력)
- **wuxing_mode**: "visible_only" / "with_hidden" / "with_branch"
- **ten_gods_mode**: "visible_only" / "main_only" / "with_hidden"
- **use_true_solar_time**: 진태양시 보정 사용 여부
- **prefer_precise_day_hour**: 정밀 계산 우선 사용

## 🧪 테스트

골든 테스트 실행:
```bash
python tests_golden.py
```

## 🌐 배포

### Render.com 배포
1. GitHub 저장소 연결
2. 환경 변수 설정:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
   - `ENV=production`
   - `ALLOWED_ORIGINS=https://yourdomain.com`

### 로컬 개발
```bash
export ENV=development
uvicorn api_server:app --reload
```

## 📝 개발 노트

- **Pydantic v2** 호환성 적용
- **타입 힌트** 완전 적용
- **에러 처리** 강화
- **입력 검증** 추가
- **보안** 강화 (CORS, debug 엔드포인트)

## 🔒 보안

- API 키는 환경 변수로 관리
- 운영 환경에서는 debug 엔드포인트 비활성화
- CORS 설정으로 허용 도메인 제한

## 📚 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Pydantic v2 마이그레이션](https://docs.pydantic.dev/latest/migration/)
- [만세력(四柱) 기초](https://ko.wikipedia.org/wiki/사주)

## 🤝 기여

프론트엔드 개발자와 협업하여 완전한 웹 서비스로 발전시킬 예정입니다.

## 📄 라이선스

이 프로젝트는 개인 학습 및 상업적 목적으로 사용됩니다.
