# tests_golden.py
# 간단 골든테스트: 우리가 신뢰하는 몇 가지 입력에 대해
# 연/월/일/시주 결과가 기대값과 정확히 일치하는지 검사합니다.

import sys
import os
from bazi_step6 import compute_bazi_payload

# 콘솔 출력 인코딩 설정
if sys.platform == "win32":
    # Windows에서 UTF-8 출력 강제
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass

def fourpillars_tuple(payload: dict) -> tuple[str,str,str,str]:
    p = payload["pillars"]
    y = p["year"]["gan"] + p["year"]["zhi"]
    m = p["month"]["gan"] + p["month"]["zhi"]
    d = p["day"]["gan"] + p["day"]["zhi"]
    h = p["hour"]["gan"] + p["hour"]["zhi"]
    return (y,m,d,h)

def check_case(name: str, birth_local: str, tz: str, sex: str,
               calendar="solar", lunar_is_leap_month=False,
               prefer_precise_day_hour=True, use_true_solar_time=True, longitude_deg=None,
               expected: tuple[str,str,str,str] = None):
    try:
        payload = compute_bazi_payload(
            birth_local, tz, sex=sex,
            calendar=calendar,
            lunar_is_leap_month=lunar_is_leap_month,
            day_boundary_policy="midnight",
            prefer_precise_day_hour=prefer_precise_day_hour,
            use_true_solar_time=use_true_solar_time,
            longitude_deg=longitude_deg,
        )
        got = fourpillars_tuple(payload)
        if expected is None:
            print(f"[{name}] => {got}  (기대값 미지정)")
            return True
        ok = (got == expected)
        status = "PASS ✅" if ok else "FAIL ❌"
        print(f"[{name}] {status}  got={got}, expected={expected}")
        return ok
    except Exception as e:
        print(f"[{name}] ERROR ❌  {e}")
        return False

def main():
    print("=== 만세력 계산기 골든 테스트 ===\n")
    all_ok = True

    # ✅ 골든케이스 1: (국내 서비스 기준과 일치 확인용)
    # 1986-10-18 21:00, 서울, 남성, 양력, LMT=ON → 시주가 丙戌(병술) 이어야 함
    all_ok &= check_case(
        "KR-1986-10-18 21:00 M",
        "1986-10-18 21:00:00", "Asia/Seoul", "M",
        calendar="solar",
        prefer_precise_day_hour=True,
        use_true_solar_time=True,        # LMT 보정
        longitude_deg=126.9784,          # 서울 경도
        expected=("丙寅","戊戌","乙未","丙戌")
    )

    # ✅ 골든케이스 2: (이전에 예제로 썼던 값)
    all_ok &= check_case(
        "KR-1995-08-26 16:00 F",
        "1995-08-26 16:00:00", "Asia/Seoul", "F",
        calendar="solar",
        prefer_precise_day_hour=True,
        use_true_solar_time=True,
        longitude_deg=126.9784,
        expected=("乙亥","甲申","甲寅","庚申")
    )

    # 👍 라벨/부가필드가 제대로 들어오는지만 확인(정답은 고정 안 함)
    ok3 = check_case(
        "Label-Only-Check",
        "2001-03-05 00:30:00", "Asia/Seoul", "M",
        calendar="solar",
        prefer_precise_day_hour=True,
        use_true_solar_time=True,
        longitude_deg=126.9784,
        expected=None  # 기대값 미지정 → 그냥 결과만 출력
    )
    all_ok &= ok3

    print("\n=== SUMMARY ===")
    if all_ok:
        print("🎉 ALL PASS ✅ - 모든 테스트가 통과했습니다!")
    else:
        print("❌ SOME FAIL - 일부 테스트가 실패했습니다.")
        print("→ FAIL 난 케이스를 기준으로 코드 변경분을 되돌리거나 원인 분석하세요.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
