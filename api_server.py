# api_server.py
import os
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ★ 추가: 정적 파일/HTML 응답
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# =========================
# ENV 로딩 (dotenv + 수동 폴백)
# =========================
def _manual_load_env_from(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    changed = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ.setdefault(k, v)
                changed = True
    if changed:
        print(f"[env] manually loaded from: {path}")
    return changed

def _ensure_env_loaded():
    try:
        from dotenv import load_dotenv  # pip install python-dotenv
        for candidate in (".env", "OPENAI_API_KEY.env", "OPENAI_API_KEY"):
            p = Path(__file__).with_name(candidate)
            if p.exists():
                load_dotenv(dotenv_path=p, override=True, encoding="utf-8")
                print(f"[env] dotenv loaded: {p}")
                break
    except Exception as e:
        print(f"[env] python-dotenv not available or failed: {e}")

    if not os.environ.get("OPENAI_API_KEY"):
        for candidate in (".env", "OPENAI_API_KEY.env", "OPENAI_API_KEY", "OPENAI_API_KEY.txt"):
            p = Path(__file__).with_name(candidate)
            if p.exists():
                if _manual_load_env_from(p):
                    break

    print(f"[env] has OPENAI_API_KEY? {bool(os.environ.get('OPENAI_API_KEY'))}")

_ensure_env_loaded()

# --- 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bazi_api")

# 당신이 만든 모듈에서 함수 import
from bazi_step6 import compute_bazi_payload, generate_text_report

app = FastAPI(title="Bazi API", version="1.4")

# CORS (초기 개발 단계에서는 * 허용, 운영에서는 도메인 제한 권장)
def get_cors_origins():
    """환경에 따른 CORS 설정"""
    env = os.environ.get("ENV", "development")
    if env == "production":
        # 운영 환경에서는 특정 도메인만 허용
        allowed_origins = os.environ.get("ALLOWED_ORIGINS", "https://yourdomain.com").split(",")
        return [origin.strip() for origin in allowed_origins]
    else:
        # 개발 환경에서는 모든 도메인 허용
        return ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 정적 파일/루트 페이지 서빙 추가
# =========================
PUBLIC_DIR = Path(__file__).with_name("public")

# /static/* 로 public 폴더 제공 (이미지/JS/CSS)
# check_dir=False: 폴더가 없어도 서버가 죽지 않도록(배포 중 안전)
app.mount(
    "/static",
    StaticFiles(directory=str(PUBLIC_DIR), html=False, check_dir=False),
    name="static",
)

# 루트(/)에서 index.html 제공
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    # index.html이 없을 때 친절한 안내
    html = """
    <html><body style="font-family:system-ui">
      <h2>Index not found</h2>
      <p><code>public/index.html</code> 파일이 없습니다.</p>
      <ul>
        <li>리포 루트에 <code>public/</code> 폴더를 만들고 <code>index.html</code>을 넣으세요.</li>
        <li>정적 파일은 <code>/static/파일명</code> 으로 접근 가능합니다.</li>
        <li>API는 <code>/api/...</code>, 문서는 <code>/docs</code> 입니다.</li>
      </ul>
    </body></html>
    """
    return HTMLResponse(content=html, status_code=200)

# === AI 해석 생성기 (OpenAI v1) ===
def render_ai_report(payload: dict, style: str | None = None) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; skipping AI report.")
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        try:
            from openai import OpenAI  # openai>=1.x
        except Exception:
            logger.exception("OpenAI SDK import failed. Install with: pip install -U openai")
            return None

        client = OpenAI(api_key=api_key)

        system = (
            "You are a Korean Saju (BaZi) interpreter. "
            "You receive a fully computed BaZi payload as JSON (do NOT recalculate). "
            "Explain clearly in Korean, kindly and responsibly. "
            "Use suggestions rather than absolutes. "
            "Sections: 1) 총평 2) 오행 3) 십성 4) 대운 5) 실천 팁. "
            "Always end with a short disclaimer. Keep it concise for MVP."
        )
        if style:
            system += f" Prefer this tone/style: {style}"

        user = json.dumps(payload, ensure_ascii=False)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception:
        logger.exception("AI report generation failed")
        return None

# -------------------------
# 요청 스키마: /api/bazi/calc
# -------------------------
class CalcRequest(BaseModel):
    birth_local: str                 # "YYYY-MM-DD HH:MM:SS"
    tz: str                          # 예: "Asia/Seoul"
    sex: str = "F"                   # "M" or "F"
    include_text_report: bool = True

    # 달력/경계
    calendar: str = "solar"          # "solar" | "lunar"
    lunar_is_leap_month: bool = False
    day_boundary_policy: str = "midnight"  # "midnight" | "zi23" | "zi-split"

    # 정밀/태양시 옵션
    prefer_precise_day_hour: bool = True   # EightChar로 일/시 정밀 산출
    use_true_solar_time: bool = False      # 진태양시(LMT) 보정
    longitude_deg: float | None = None     # LMT 보정 시 경도(예: 서울 126.9784)

    # ★ 집계/표시 신규 옵션
    wuxing_mode: str = "with_hidden"       # "visible_only" | "with_hidden" | "with_branch" | "hidden_main_only"
    ten_gods_mode: str = "with_hidden"     # "visible_only" | "main_only" | "with_hidden"
    hidden_order: str = "std"              # "std" | "cmj" | "mjc"

    model_config = ConfigDict(
        schema_extra={
            "example": {
                "birth_local": "1982-05-25 20:00:00",
                "tz": "Asia/Seoul",
                "sex": "M",
                "include_text_report": True,

                "calendar": "solar",
                "lunar_is_leap_month": False,
                "day_boundary_policy": "midnight",

                "prefer_precise_day_hour": True,
                "use_true_solar_time": True,
                "longitude_deg": 126.9784,

                "wuxing_mode": "with_branch",
                "ten_gods_mode": "main_only",
                "hidden_order": "cmj"
            }
        }
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/bazi/calc")
def calc(req: CalcRequest):
    # --- 편의: LMT 요청인데 경도 미지정 & 서울 표준시면 자동 보완 ---
    if req.use_true_solar_time and req.longitude_deg is None and req.tz == "Asia/Seoul":
        req.longitude_deg = 126.9784  # 서울 경도(대략)

    try:
        payload = compute_bazi_payload(
            req.birth_local,
            req.tz,
            sex=req.sex,
            calendar=req.calendar,
            lunar_is_leap_month=req.lunar_is_leap_month,
            day_boundary_policy=req.day_boundary_policy,
            prefer_precise_day_hour=req.prefer_precise_day_hour,
            use_true_solar_time=req.use_true_solar_time,
            longitude_deg=req.longitude_deg,
            # 신규 옵션
            wuxing_mode=req.wuxing_mode,
            ten_gods_mode=req.ten_gods_mode,
            hidden_order=req.hidden_order,
        )
    except TypeError:
        # 구버전 호환(추가 인자 미지원 시)
        payload = compute_bazi_payload(
            req.birth_local,
            req.tz,
            sex=req.sex,
            calendar=req.calendar,
            lunar_is_leap_month=req.lunar_is_leap_month,
            day_boundary_policy=req.day_boundary_policy,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    if req.include_text_report:
        try:
            report = generate_text_report(payload)
        except Exception as e:
            report = f"(report generation failed: {e})"
        return {"ok": True, "payload": payload, "text_report": report}
    else:
        return {"ok": True, "payload": payload}

# 브라우저 GET 테스트용 (주소창/링크로 확인하기 편함)
@app.get("/api/bazi/calc")
def calc_get(
    birth_local: str,
    tz: str,
    sex: str = "F",
    include_text_report: bool = True,
    calendar: str = "solar",
    lunar_is_leap_month: bool = False,
    day_boundary_policy: str = "midnight",
    prefer_precise_day_hour: bool = True,
    use_true_solar_time: bool = False,
    longitude_deg: float | None = None,
    wuxing_mode: str = "with_hidden",
    ten_gods_mode: str = "with_hidden",
    hidden_order: str = "std",
):
    req = CalcRequest(
        birth_local=birth_local,
        tz=tz,
        sex=sex,
        include_text_report=include_text_report,
        calendar=calendar,
        lunar_is_leap_month=lunar_is_leap_month,
        day_boundary_policy=day_boundary_policy,
        prefer_precise_day_hour=prefer_precise_day_hour,
        use_true_solar_time=use_true_solar_time,
        longitude_deg=longitude_deg,
        wuxing_mode=wuxing_mode,
        ten_gods_mode=ten_gods_mode,
        hidden_order=hidden_order,
    )
    return calc(req)

# -------------------------
# 분석(폼 입력)용 단일 엔드포인트
# -------------------------
class AnalyzeRequest(BaseModel):
    # 날짜/시간(폼 입력 그대로)
    year: int; month: int; day: int
    hour: int; minute: int = 0; second: int = 0
    sex: Annotated[str, Field(pattern="^(M|F|m|f)$")] = "F"
    # 위치/시간대
    tz: str | None = "Asia/Seoul"
    # 달력/옵션
    calendar: str = "solar"
    lunar_is_leap_month: bool = False
    day_boundary_policy: str = "midnight"
    prefer_precise_day_hour: bool = True
    use_true_solar_time: bool = True
    longitude_deg: float | None = None
    include_ai: bool = True
    ai_style: str | None = None

    # ★ 신규 옵션
    wuxing_mode: str = "with_hidden"
    ten_gods_mode: str = "with_hidden"
    hidden_order: str = "std"

    model_config = ConfigDict(
        schema_extra={
            "example": {
                "year": 1982, "month": 5, "day": 25,
                "hour": 20, "minute": 0, "second": 0,
                "sex": "M",
                "tz": "Asia/Seoul",

                "calendar": "solar",
                "lunar_is_leap_month": False,
                "day_boundary_policy": "midnight",

                "prefer_precise_day_hour": True,
                "use_true_solar_time": True,
                "longitude_deg": 126.9784,

                "wuxing_mode": "with_branch",
                "ten_gods_mode": "main_only",
                "hidden_order": "cmj",

                "include_ai": True,
                "ai_style": "친절하고 간결하게"
            }
        }
    )

def _auto_lon_if_needed(tz: str | None, lon: float | None):
    if lon is not None:
        return lon
    return 126.9784 if tz == "Asia/Seoul" else None

def _validate_date_time(year: int, month: int, day: int, hour: int, minute: int, second: int) -> None:
    """날짜/시간 입력값 검증"""
    if not (1900 <= year <= 2100):
        raise ValueError("년도는 1900-2100 범위여야 합니다")
    if not (1 <= month <= 12):
        raise ValueError("월은 1-12 범위여야 합니다")
    if not (1 <= day <= 31):
        raise ValueError("일은 1-31 범위여야 합니다")
    if not (0 <= hour <= 23):
        raise ValueError("시는 0-23 범위여야 합니다")
    if not (0 <= minute <= 59):
        raise ValueError("분은 0-59 범위여야 합니다")
    if not (0 <= second <= 59):
        raise ValueError("초는 0-59 범위여야 합니다")

def _validate_timezone(tz: str) -> None:
    """타임존 유효성 검증"""
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
    except Exception:
        raise ValueError(f"유효하지 않은 타임존입니다: {tz}")

@app.post("/api/bazi/analyze")
def analyze(req: AnalyzeRequest):
    # 1) 입력값 검증
    try:
        _validate_date_time(req.year, req.month, req.day, req.hour, req.minute, req.second)
        if req.tz:
            _validate_timezone(req.tz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"입력값 오류: {e}")
    
    # 2) 폼 → birth_local 조립
    birth_local = f"{req.year:04d}-{req.month:02d}-{req.day:02d} {req.hour:02d}:{req.minute:02d}:{req.second:02d}"
    tz = req.tz or "Asia/Seoul"

    # 3) LMT 경도 자동 보완 (국내 기본 ON)
    lon = _auto_lon_if_needed(tz, req.longitude_deg) if req.use_true_solar_time else None

    # 4) 계산
    try:
        payload = compute_bazi_payload(
            birth_local, tz, sex=req.sex,
            calendar=req.calendar,
            lunar_is_leap_month=req.lunar_is_leap_month,
            day_boundary_policy=req.day_boundary_policy,
            prefer_precise_day_hour=req.prefer_precise_day_hour,
            use_true_solar_time=req.use_true_solar_time,
            longitude_deg=lon,
            # 신규 옵션 전달
            wuxing_mode=req.wuxing_mode,
            ten_gods_mode=req.ten_gods_mode,
            hidden_order=req.hidden_order,
        )
    except TypeError:
        payload = compute_bazi_payload(
            birth_local, tz, sex=req.sex,
            calendar=req.calendar,
            lunar_is_leap_month=req.lunar_is_leap_month,
            day_boundary_policy=req.day_boundary_policy,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"사주 계산 오류: {e}")

    # 5) 기본 해석(템플릿)
    try:
        text_report = generate_text_report(payload)
    except Exception as e:
        text_report = f"(report generation failed: {e})"

    # 6) AI 해석(키 없으면 자동으로 None)
    ai_report = render_ai_report(payload, req.ai_style) if req.include_ai else None

    return {
        "ok": True,
        "payload": payload,
        "text_report": text_report,
        "ai_report": ai_report,
        "ai_model": os.environ.get("OPENAI_MODEL") if ai_report else None
    }

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)

# 운영 환경에서는 debug 엔드포인트 비활성화
if os.environ.get("ENV") != "production":
    @app.get("/debug/env")
    def debug_env():
        key = os.environ.get("OPENAI_API_KEY")
        return {
            "has_api_key": bool(key),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "key_preview": (key[:7] + "..." if key else None)
        }
else:
    @app.get("/debug/env")
    def debug_env():
        return {"message": "Debug endpoint disabled in production"}
