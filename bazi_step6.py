# bazi_step6.py
# Step 6+: 지장간 + 오행 + 십성 + 대운 + 정밀절기 + 입력옵션(음력/경계정책)
# 0.14 — selectable wuxing/ten-gods modes + hidden-stems order + selected_counts

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json

ISO_FMT = "%Y-%m-%d %H:%M:%S"

# --- precise engine (lunar-python) ---
try:
    from lunar_python import Solar  # pip install lunar-python
    HAS_LUNARPY = True
except Exception:
    HAS_LUNARPY = False

# -------------------------
# 공통 데이터 구조
# -------------------------
@dataclass
class BirthInput:
    birth_local: str
    tz: str
    location_name: str | None = None
    lat: float | None = None
    lng: float | None = None

@dataclass
class NormalizedTime:
    local_iso: str
    utc_iso: str
    jdn: int
    jd: float

@dataclass
class FourPillars:
    year_gan: str | None; year_zhi: str | None
    month_gan: str | None; month_zhi: str | None
    day_gan: str | None; day_zhi: str | None
    hour_gan: str | None; hour_zhi: str | None

# -------------------------
# 1단계: 시간 정규화
# -------------------------
def parse_local_dt(birth_local: str, tz_str: str) -> datetime:
    dt_naive = datetime.strptime(birth_local, ISO_FMT)
    tz = ZoneInfo(tz_str)
    return dt_naive.replace(tzinfo=tz)

def to_utc(dt_local: datetime) -> datetime:
    return dt_local.astimezone(timezone.utc)

def gregorian_to_jd(dt_utc: datetime) -> tuple[int, float]:
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day
    frac = (dt_utc.hour + (dt_utc.minute/60) + (dt_utc.second/3600) + (dt_utc.microsecond/3_600_000_000)) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + (A // 4)
    jd0 = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    jd = jd0 + frac
    jdn = int(jd + 0.5)
    return jdn, jd

def normalize_time(inp: BirthInput) -> NormalizedTime:
    dt_local = parse_local_dt(inp.birth_local, inp.tz)
    dt_utc = to_utc(dt_local)
    jdn, jd = gregorian_to_jd(dt_utc)
    return NormalizedTime(
        local_iso=dt_local.isoformat(),
        utc_iso=dt_utc.isoformat().replace("+00:00", "Z"),
        jdn=jdn,
        jd=round(jd, 6),
    )

# -------------------------
# 2단계: 시지(時間의 地支)
# -------------------------
ZHI_LIST = [
    ("子","자"), ("丑","축"), ("寅","인"), ("卯","묘"),
    ("辰","진"), ("巳","사"), ("午","오"), ("未","미"),
    ("申","신"), ("酉","유"), ("戌","술"), ("亥","해")
]
GAN_LIST = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]

# ---- 표시용 라벨 매핑 (보기 전용) ----
GAN_TO_KOR = {
    "甲":"갑","乙":"을","丙":"병","丁":"정","戊":"무",
    "己":"기","庚":"경","辛":"신","壬":"임","癸":"계",
}
ZHI_TO_KOR = {han: kor for (han, kor) in ZHI_LIST}
WUXING_KEY_LABEL = {
    "wood":"木(목)","fire":"火(화)","earth":"土(토)","metal":"金(금)","water":"水(수)",
}

def label_gan(stem: str | None) -> str | None:
    if not stem: return None
    return f"{stem}({GAN_TO_KOR.get(stem, '?')})"

def label_zhi(branch: str | None) -> str | None:
    if not branch: return None
    return f"{branch}({ZHI_TO_KOR.get(branch, '?')})"

def hour_to_zhi_index(local_dt: datetime) -> int:
    # 자시: 23:00~00:59
    h = local_dt.hour
    if h in (23, 0): return 0
    return (h + 1) // 2

def calc_hour_branch(local_dt: datetime) -> tuple[str, str, int]:
    idx = hour_to_zhi_index(local_dt)
    zhi_han, zhi_kor = ZHI_LIST[idx]
    return zhi_han, zhi_kor, idx

# -------------------------
# 3단계: 일주(일간·일지)
# -------------------------
ANCHOR_LOCAL_DATE = "1984-02-02 00:00:00"  # 甲子日 (로컬 기준, 가정)
ANCHOR_TZ = "Asia/Seoul"

def local_midnight(dt_local: datetime) -> datetime:
    return dt_local.replace(hour=0, minute=0, second=0, microsecond=0)

def sexagenary_index_from_days(delta_days: int) -> int:
    return delta_days % 60

def split_sexagenary_index(idx60: int) -> tuple[int, int]:
    stem_idx = idx60 % 10
    branch_idx = idx60 % 12
    return stem_idx, branch_idx

def calc_day_pillar(local_dt: datetime) -> tuple[str, str, int, int]:
    birth_midnight_local = local_midnight(local_dt)
    birth_midnight_utc = to_utc(birth_midnight_local)

    anchor_local = parse_local_dt(ANCHOR_LOCAL_DATE, ANCHOR_TZ)
    anchor_midnight_local = local_midnight(anchor_local)
    anchor_midnight_utc = to_utc(anchor_midnight_local)

    delta = anchor_midnight_utc - birth_midnight_utc
    delta_days = int(delta.total_seconds() // 86400)  # anchor - birth
    idx60 = sexagenary_index_from_days((-delta_days) % 60)

    stem_idx, branch_idx = split_sexagenary_index(idx60)
    return GAN_LIST[stem_idx], ZHI_LIST[branch_idx][0], stem_idx, branch_idx

# -------------------------
# 4단계: 시간간(時間干)
# -------------------------
DAY_STEM_TO_ZI_HOUR_STEM = {
    "甲": "甲", "己": "甲",
    "乙": "丙", "庚": "丙",
    "丙": "戊", "辛": "戊",
    "丁": "庚", "壬": "庚",
    "戊": "壬", "癸": "壬",
}
def stem_index(stem: str) -> int:
    return GAN_LIST.index(stem)
def next_stem(stem_idx: int, step: int) -> str:
    return GAN_LIST[(stem_idx + step) % 10]
def calc_hour_stem(day_stem: str, hour_branch_index: int) -> str:
    zi_stem = DAY_STEM_TO_ZI_HOUR_STEM[day_stem]
    base_idx = stem_index(zi_stem)
    return next_stem(base_idx, hour_branch_index)

# -------------------------
# 5단계: 연주·월주 (정밀 절기 지원)
# -------------------------
def approx_is_after_lichun(local_dt: datetime) -> bool:
    y = local_dt.year
    lichun = local_dt.replace(year=y, month=2, day=4, hour=0, minute=0, second=0, microsecond=0)
    return local_dt >= lichun

def _split_ganzhi_pair(val: str):
    if isinstance(val, str) and len(val) >= 2:
        return val[0], val[1]
    return None

def _get_part_from_eightchar(ec_obj, part_names: list[str]) -> tuple[str|None, str|None]:
    """
    lunar_python 버전별로 이름이 다를 수 있으므로 가능한 모든 조합을 시도.
    우선순위: get{Part}GanZhi() → get{Part}Gan()/get{Part}Zhi() → get{Part}() (결합 문자열)
    part_names 예: ["Year"], ["Month"], ["Day"], ["Time","Hour"]
    """
    def _split_gz(s):
        if isinstance(s, str) and len(s) >= 2:
            return s[0], s[1]
        return (None, None)

    for part in part_names:
        m = getattr(ec_obj, f"get{part}GanZhi", None)
        if callable(m):
            try:
                gz = m()
                g, z = _split_gz(gz)
                if g and z:
                    return g, z
            except Exception:
                pass

        get_gan = getattr(ec_obj, f"get{part}Gan", None)
        get_zhi = getattr(ec_obj, f"get{part}Zhi", None)
        if callable(get_gan) and callable(get_zhi):
            try:
                gan = get_gan(); zhi = get_zhi()
                if isinstance(gan, str) and isinstance(zhi, str):
                    return gan, zhi
            except Exception:
                pass

        get_both = getattr(ec_obj, f"get{part}", None)
        if callable(get_both):
            try:
                val = get_both()
                g, z = _split_gz(val)
                if g and z:
                    return g, z
            except Exception:
                pass

    return None, None

def calc_year_month_pillars_precise(local_dt) -> tuple[tuple[str,str] | None, tuple[str,str] | None]:
    if not HAS_LUNARPY:
        return None, None
    try:
        sol = Solar.fromYmdHms(local_dt.year, local_dt.month, local_dt.day,
                               local_dt.hour, local_dt.minute, local_dt.second)
        lunar = sol.getLunar()
        get_ec = getattr(lunar, "getEightChar", None)
        ec = get_ec() if callable(get_ec) else getattr(lunar, "eightChar", None)
        if ec is None:
            return None, None
        yg, yz = _get_part_from_eightchar(ec, ["Year"])
        mg, mz = _get_part_from_eightchar(ec, ["Month"])
        if yg and yz and mg and mz:
            return (yg, yz), (mg, mz)
    except Exception:
        pass
    return None, None

def calc_year_pillar(local_dt: datetime) -> tuple[str, str]:
    if approx_is_after_lichun(local_dt):
        y_for_cycle = local_dt.year
    else:
        y_for_cycle = local_dt.year - 1
    diff = y_for_cycle - 1984  # 1984=甲子
    idx60 = diff % 60
    stem_idx, branch_idx = split_sexagenary_index(idx60)
    return GAN_LIST[stem_idx], ZHI_LIST[branch_idx][0]

APPROX_MONTH_BOUNDARIES = [
    (2, 4,  "寅"), (3, 6,  "卯"), (4, 5,  "辰"), (5, 6,  "巳"),
    (6, 6,  "午"), (7, 7,  "未"), (8, 8,  "申"), (9, 8,  "酉"),
    (10,8,  "戌"), (11,7,  "亥"), (12,7,  "子"), (1, 6,  "丑"),
]
Y_STEM_TO_YIN_MONTH_STEM = {
    "甲": "丙", "己": "丙",
    "乙": "戊", "庚": "戊",
    "丙": "庚", "辛": "庚",
    "丁": "壬", "壬": "壬",
    "戊": "甲", "癸": "甲",
}
def zhi_to_index(zhi_han: str) -> int:
    return [z for z,_ in ZHI_LIST].index(zhi_han)

def calc_month_index_and_branch(local_dt: datetime) -> tuple[int, str]:
    tz = local_dt.tzinfo
    candidates = []
    for year_offset in (-1, 0, 1):
        y = local_dt.year + year_offset
        for (m, d, zhi) in APPROX_MONTH_BOUNDARIES:
            dt = datetime(y, m, d, 0, 0, 0, tzinfo=tz)
            candidates.append((dt, zhi))
    candidates.sort(key=lambda x: x[0])
    prev = None
    for i, (dt, zhi) in enumerate(candidates):
        if dt <= local_dt:
            prev = (dt, zhi, i)
        else:
            break
    if prev is None:
        dt, zhi, i = candidates[0][0], candidates[0][1], 0
    else:
        dt, zhi, i = prev
    zhi_idx = zhi_to_index(zhi)
    yin_idx = zhi_to_index("寅")
    month_no = ((zhi_idx - yin_idx) % 12) + 1
    return month_no, zhi

def next_stem_by_add(stem_han: str, add: int) -> str:
    si = stem_index(stem_han)
    return GAN_LIST[(si + add) % 10]

def calc_month_pillar(local_dt: datetime, year_stem: str) -> tuple[str, str, int]:
    month_no, month_branch = calc_month_index_and_branch(local_dt)
    yin_start = Y_STEM_TO_YIN_MONTH_STEM[year_stem]
    month_stem = next_stem_by_add(yin_start, month_no - 1)
    return month_stem, month_branch, month_no

def find_adjacent_boundaries(local_dt: datetime) -> tuple[datetime, datetime]:
    tz = local_dt.tzinfo
    candidates = []
    for year_offset in (-1, 0, 1):
        y = local_dt.year + year_offset
        for (m, d, _zhi) in APPROX_MONTH_BOUNDARIES:
            candidates.append(datetime(y, m, d, 0, 0, 0, tzinfo=tz))
    candidates.sort()
    prev_dt = candidates[0]
    next_dt = candidates[-1]
    for i, dt in enumerate(candidates):
        if dt <= local_dt:
            prev_dt = dt
            next_dt = candidates[min(i + 1, len(candidates) - 1)]
        else:
            next_dt = dt
            break
    return prev_dt, next_dt

# -------------------------
# 11단계: 정밀 절기 경계(이전/다음) 찾기
# -------------------------
def _solar_to_local_dt(solar_obj, tz) -> datetime | None:
    try:
        y = solar_obj.getYear(); m = solar_obj.getMonth(); d = solar_obj.getDay()
        h = solar_obj.getHour(); mi = solar_obj.getMinute(); se = solar_obj.getSecond()
        return datetime(y, m, d, h, mi, se, tzinfo=tz)
    except Exception:
        pass
    try:
        s = solar_obj.toYmdHms()  # "YYYY-MM-DD HH:MM:SS"
        ymd, hms = s.split(" ")
        y, m, d = map(int, ymd.split("-"))
        h, mi, se = map(int, hms.split(":"))
        return datetime(y, m, d, h, mi, se, tzinfo=tz)
    except Exception:
        return None

def find_adjacent_boundaries_precise(local_dt: datetime) -> tuple[datetime, datetime] | None:
    if not HAS_LUNARPY:
        return None
    try:
        def collect_year(y: int):
            sol = Solar.fromYmdHms(y, local_dt.month, local_dt.day,
                                   local_dt.hour, local_dt.minute, local_dt.second)
            lun = sol.getLunar()
            get_tbl = getattr(lun, "getJieQiTable", None)
            if not callable(get_tbl): return []
            tbl = get_tbl()  # { '立春': Solar, ... }
            out = []
            for _name, s in tbl.items():
                dt = _solar_to_local_dt(s, local_dt.tzinfo)
                if dt: out.append(dt)
            return out

        dts = collect_year(local_dt.year) + collect_year(local_dt.year - 1) + collect_year(local_dt.year + 1)
        if not dts: return None
        dts = sorted(set(dts))

        prev_bd = None; next_bd = None
        for dt in dts:
            if dt <= local_dt: prev_bd = dt
            if dt > local_dt and next_bd is None: next_bd = dt

        if prev_bd is None or next_bd is None:
            more = []
            for yy in (local_dt.year - 2, local_dt.year + 2):
                more += collect_year(yy)
            dts2 = sorted(set(dts + more))
            for dt in dts2:
                if dt <= local_dt: prev_bd = dt
                if dt > local_dt and next_bd is None: next_bd = dt

        return (prev_bd, next_bd) if (prev_bd and next_bd) else None
    except Exception:
        return None

def find_adjacent_boundaries_auto(local_dt: datetime) -> tuple[datetime, datetime, str]:
    precise = find_adjacent_boundaries_precise(local_dt)
    if precise:
        return precise[0], precise[1], "precise_jieqi"
    prev_dt, next_dt = find_adjacent_boundaries(local_dt)
    return prev_dt, next_dt, "approx_jieqi"

# ---- True Solar Time (Local Mean Solar Time) 보정 ----
def adjust_to_apparent_solar_time(dt_local: datetime, longitude_deg: float) -> datetime:
    """
    경도 기반 LMT 보정(평균태양시). 표준자오선(UTC오프셋×15°)과의 차이 × 4분/도.
    * 간단 모델: 방정시(Equation of Time)는 무시 (대부분 상용 웹 서비스 기본과 동일).
    """
    tz_hours = dt_local.utcoffset().total_seconds() / 3600.0
    central_meridian = 15.0 * tz_hours
    minutes_offset = 4.0 * (longitude_deg - central_meridian)  # 분
    return dt_local + timedelta(minutes=minutes_offset)

# ---- EightChar에서 Year/Month/Day(표준시간) + Hour(옵션 LMT) 분리 취득 ----
def calc_pillars_precise_all(local_dt: datetime, use_true_solar_time: bool = False, longitude_deg: float | None = None):
    """
    EightChar를 이용해 연/월/일/시를 계산하되,
    - 연/월/일: 표준(타임존) 로컬시간으로 계산
    - 시: (옵션) LMT 보정 시간을 사용해 계산
    이렇게 분리하여 시주만 태양시 보정의 영향을 받게 한다.
    반환: ((Yg,Yz),(Mg,Mz),(Dg,Dz),(Hg,Hz), local_dt) 또는 None
    """
    if not HAS_LUNARPY:
        return None
    try:
        # ---- 1) 연/월/일: 표준 로컬시간 ----
        sol_day = Solar.fromYmdHms(local_dt.year, local_dt.month, local_dt.day,
                                   local_dt.hour, local_dt.minute, local_dt.second)
        lun_day = sol_day.getLunar()
        get_ec = getattr(lun_day, "getEightChar", None)
        ec_day = get_ec() if callable(get_ec) else getattr(lun_day, "eightChar", None)
        if ec_day is None:
            return None

        yg, yz = _get_part_from_eightchar(ec_day, ["Year"])
        mg, mz = _get_part_from_eightchar(ec_day, ["Month"])
        dg, dz = _get_part_from_eightchar(ec_day, ["Day"])
        if not (yg and yz and mg and mz and dg and dz):
            return None

        # ---- 2) 시: (옵션) LMT 보정 시간으로 ----
        dt_hour = local_dt
        if use_true_solar_time and (longitude_deg is not None):
            dt_hour = adjust_to_apparent_solar_time(local_dt, longitude_deg)

        sol_hour = Solar.fromYmdHms(dt_hour.year, dt_hour.month, dt_hour.day,
                                    dt_hour.hour, dt_hour.minute, dt_hour.second)
        lun_hour = sol_hour.getLunar()
        get_ec2 = getattr(lun_hour, "getEightChar", None)
        ec_hour = get_ec2() if callable(get_ec2) else getattr(lun_hour, "eightChar", None)
        if ec_hour is None:
            return None

        hg, hz = _get_part_from_eightchar(ec_hour, ["Time","Hour"])
        if not (hg and hz):
            return None

        return ((yg, yz), (mg, mz), (dg, dz), (hg, hz), local_dt)
    except Exception:
        return None

# -------------------------
# 6단계: 지장간 + 오행
# -------------------------
# (표준 순서: 본기-중기-여기)
HIDDEN_STEMS_BY_ZHI = {
    "子": ["癸"],
    "丑": ["己","癸","辛"],
    "寅": ["甲","丙","戊"],
    "卯": ["乙"],
    "辰": ["戊","乙","癸"],
    "巳": ["丙","戊","庚"],
    "午": ["丁","己"],
    "未": ["己","乙","丁"],
    "申": ["庚","壬","戊"],
    "酉": ["辛"],
    "戌": ["戊","辛","丁"],
    "亥": ["壬","甲"],
}
STEM_TO_WUXING = {
    "甲":"wood","乙":"wood",
    "丙":"fire","丁":"fire",
    "戊":"earth","己":"earth",
    "庚":"metal","辛":"metal",
    "壬":"water","癸":"water",
}
# 각 지지의 '본기' 오행(많은 서비스에서 가지수 4를 맞출 때 사용)
BRANCH_TO_WUXING_MAIN = {
    "子":"water","丑":"earth","寅":"wood","卯":"wood",
    "辰":"earth","巳":"fire","午":"fire","未":"earth",
    "申":"metal","酉":"metal","戌":"earth","亥":"water",
}

def _reorder_hidden_list(lst: list[str], order: str) -> list[str]:
    n = len(lst)
    if n <= 1 or order == "std":
        return lst[:]
    if order == "cmj":
        # 중-여-본 (예: 戌 [戊,辛,丁] → [辛,丁,戊])
        if   n == 3: idx = [1,2,0]
        elif n == 2: idx = [1,0]
        else: idx = [0]
        return [lst[i] for i in idx]
    if order == "mjc":
        # 여-본-중 (예: 戌 [戊,辛,丁] → [丁,戊,辛])
        if   n == 3: idx = [2,0,1]
        elif n == 2: idx = [1,0]
        else: idx = [0]
        return [lst[i] for i in idx]
    return lst[:]

def hidden_stems_for_branch(zhi: str, order: str = "std") -> list[str]:
    base = HIDDEN_STEMS_BY_ZHI.get(zhi, [])
    return _reorder_hidden_list(base, order)

def compute_hidden_stems(fp: FourPillars, order: str = "std") -> dict:
    return {
        "year":  hidden_stems_for_branch(fp.year_zhi,  order) if fp.year_zhi  else [],
        "month": hidden_stems_for_branch(fp.month_zhi, order) if fp.month_zhi else [],
        "day":   hidden_stems_for_branch(fp.day_zhi,   order) if fp.day_zhi   else [],
        "hour":  hidden_stems_for_branch(fp.hour_zhi,  order) if fp.hour_zhi  else [],
    }

def count_wuxing(stems: list[str]) -> dict:
    counts = {"wood":0,"fire":0,"earth":0,"metal":0,"water":0}
    for s in stems:
        w = STEM_TO_WUXING.get(s)
        if w: counts[w] += 1
    return counts

def compute_wuxing_distribution(fp: FourPillars, *, mode: str = "with_hidden", hidden_order: str = "std") -> dict:
    """
    오행 집계 모드:
      - visible_only   : 4개(연간/월간/일간/시간)
      - with_hidden    : 4개 가간 + 각 지지의 모든 지장간
      - with_branch    : 4개 가간 + 4개 '지지 본기 오행'(BRANCH_TO_WUXING_MAIN)
      - hidden_main_only: 지장간 '본기' 4개만 (가간 제외)
    반환 딕셔너리에는 모든 방식의 참조값과 'selected_counts'가 함께 들어감.
    """
    # 4개의 '가간' (필수)
    visible_stems = [x for x in [fp.year_gan, fp.month_gan, fp.day_gan, fp.hour_gan] if x]

    # 지장간(표시 순서 옵션 반영)
    hidden_map = compute_hidden_stems(fp, order=hidden_order)
    # '본기'(표준 표에서 첫 항목)를 가져올 때는 표준 순서를 기준으로 삼는다(표시 순서와 무관).
    main_hidden_by_branch_std = {
        "year": (HIDDEN_STEMS_BY_ZHI.get(fp.year_zhi,  []) or [None])[0] if fp.year_zhi  else None,
        "month":(HIDDEN_STEMS_BY_ZHI.get(fp.month_zhi, []) or [None])[0] if fp.month_zhi else None,
        "day":  (HIDDEN_STEMS_BY_ZHI.get(fp.day_zhi,   []) or [None])[0] if fp.day_zhi   else None,
        "hour": (HIDDEN_STEMS_BY_ZHI.get(fp.hour_zhi,  []) or [None])[0] if fp.hour_zhi  else None,
    }
    # branch 본기 오행 4개
    branch_main_elems = []
    for z in [fp.year_zhi, fp.month_zhi, fp.day_zhi, fp.hour_zhi]:
        if z:
            branch_main_elems.append(BRANCH_TO_WUXING_MAIN.get(z))

    # 다양한 참조 집계
    counts_visible_only = count_wuxing(visible_stems)
    all_hidden_flat = []
    for arr in hidden_map.values():
        all_hidden_flat.extend(arr)
    counts_with_hidden = count_wuxing(visible_stems + all_hidden_flat)
    counts_branch_only = {"wood":0,"fire":0,"earth":0,"metal":0,"water":0}
    for e in branch_main_elems:
        if e: counts_branch_only[e] += 1
    counts_hidden_main_only = count_wuxing([s for s in main_hidden_by_branch_std.values() if s])

    # 선택 모드
    if mode == "visible_only":
        selected = counts_visible_only
    elif mode == "with_hidden":
        selected = counts_with_hidden
    elif mode == "with_branch":
        # 많은 상용 서비스가 '가간 4 + 본기 4'를 더해 8개로 집계
        tmp = {"wood":0,"fire":0,"earth":0,"metal":0,"water":0}
        for k in tmp: tmp[k] = counts_visible_only[k] + counts_branch_only[k]
        selected = tmp
    elif mode == "hidden_main_only":
        selected = counts_hidden_main_only
    else:
        selected = counts_with_hidden

    return {
        "mode": mode,
        "visible_only": counts_visible_only,
        "with_hidden": counts_with_hidden,
        "branch_only": counts_branch_only,            # 4개 지지 본기만
        "hidden_main_only": counts_hidden_main_only,  # 지장간 본기만(4)
        "hidden_by_pillar": hidden_map,               # 표시 순서 반영됨
        "selected_counts": selected,                  # ← 화면/요약에 사용할 값
    }

# -------------------------
# 7단계: 십성(十神) 계산
# -------------------------
WUXING_SHENG = {"wood":"fire","fire":"earth","earth":"metal","metal":"water","water":"wood"}
WUXING_KE    = {"wood":"earth","earth":"water","water":"fire","fire":"metal","metal":"wood"}
YANG_STEMS = {"甲","丙","戊","庚","壬"}
def is_yang_stem(stem: str) -> bool:
    return stem in YANG_STEMS
def stem_element(stem: str) -> str:
    return STEM_TO_WUXING[stem]
TEN_GOD_LABELS_KO = {
    "BIJIAN":"비견","JECAI":"겁재",
    "SHISHEN":"식신","SHANGGUAN":"상관",
    "PIANCAI":"편재","ZHENGCAI":"정재",
    "QISHA":"편관","ZHENGGUAN":"정관",
    "PIANYIN":"편인","ZHENGYIN":"정인",
}
def ten_god_for(day_stem: str, target_stem: str) -> str:
    dm = stem_element(day_stem)
    tg = stem_element(target_stem)
    same_polarity = (is_yang_stem(day_stem) == is_yang_stem(target_stem))
    if tg == dm:
        return TEN_GOD_LABELS_KO["BIJIAN"] if same_polarity else TEN_GOD_LABELS_KO["JECAI"]
    if WUXING_SHENG[dm] == tg:
        return TEN_GOD_LABELS_KO["SHISHEN"] if same_polarity else TEN_GOD_LABELS_KO["SHANGGUAN"]
    if WUXING_KE[dm] == tg:
        return TEN_GOD_LABELS_KO["PIANCAI"] if same_polarity else TEN_GOD_LABELS_KO["ZHENGCAI"]
    if WUXING_KE[tg] == dm:
        return TEN_GOD_LABELS_KO["QISHA"] if same_polarity else TEN_GOD_LABELS_KO["ZHENGGUAN"]
    if WUXING_SHENG[tg] == dm:
        return TEN_GOD_LABELS_KO["PIANYIN"] if same_polarity else TEN_GOD_LABELS_KO["ZHENGYIN"]
    return "?"

def compute_ten_gods_visible(fp: FourPillars) -> dict:
    dm = fp.day_gan
    out = {}
    for key, stem in [("year", fp.year_gan), ("month", fp.month_gan), ("day", fp.day_gan), ("hour", fp.hour_gan)]:
        out[key] = ten_god_for(dm, stem) if (dm and stem) else None
    return out

def compute_ten_gods_counts(fp: FourPillars, *, mode: str = "with_hidden") -> dict:
    """
    십성 집계 모드:
      - visible_only : 가간(연/월/일/시) 4개만
      - main_only    : 가간 4 + 각 지지의 '본기'(1개씩)만 추가 → 총 8개
      - with_hidden  : 가간 4 + 각 지지의 지장간 전부 추가
    """
    dm = fp.day_gan
    counts = {label: 0 for label in TEN_GOD_LABELS_KO.values()}
    if not dm:
        return counts

    # 1) 가간 4개
    for stem in [fp.year_gan, fp.month_gan, fp.day_gan, fp.hour_gan]:
        if stem:
            counts[ten_god_for(dm, stem)] += 1

    # 2) 숨은 것 처리
    if mode == "visible_only":
        return counts

    if mode == "main_only":
        # 각 지지의 '본기'(표준 순서 첫 항목)만
        mains = []
        for z in [fp.year_zhi, fp.month_zhi, fp.day_zhi, fp.hour_zhi]:
            if z:
                main = (HIDDEN_STEMS_BY_ZHI.get(z, []) or [None])[0]
                if main:
                    mains.append(main)
        for s in mains:
            counts[ten_god_for(dm, s)] += 1
        return counts

    # 'with_hidden' : 전부 추가
    for z in [fp.year_zhi, fp.month_zhi, fp.day_zhi, fp.hour_zhi]:
        for s in (HIDDEN_STEMS_BY_ZHI.get(z, []) if z else []):
            counts[ten_god_for(dm, s)] += 1
    return counts

# -------------------------
# 8단계 준비: 60갑자 순환
# -------------------------
def build_sexagenary_cycle():
    pairs = []
    si, zi = 0, 0
    for _ in range(60):
        pairs.append((GAN_LIST[si], ZHI_LIST[zi][0]))
        si = (si + 1) % 10
        zi = (zi + 1) % 12
    return pairs
SEXAGENARY_60 = build_sexagenary_cycle()
def pillar_to_index(stem: str, zhi: str) -> int:
    for idx, (gs, zs) in enumerate(SEXAGENARY_60):
        if gs == stem and zs == zhi:
            return idx
    raise ValueError(f"Invalid pillar combo: {stem}{zhi}")

# -------------------------
# 8단계: 대운(大運)
# -------------------------
def luck_direction(year_stem: str, sex: str) -> int:
    yang = is_yang_stem(year_stem)
    if sex.upper().startswith("M"):
        return +1 if yang else -1
    else:
        return -1 if yang else +1

def calc_luck_start_age(local_dt: datetime, direction: int) -> tuple[float, datetime, datetime, str]:
    prev_bd, next_bd, policy = find_adjacent_boundaries_auto(local_dt)
    if direction == +1:
        diff_days = (next_bd - local_dt).total_seconds() / 86400.0
    else:
        diff_days = (local_dt - prev_bd).total_seconds() / 86400.0
    start_age = round(diff_days / 3.0, 1)
    return start_age, prev_bd, next_bd, policy

def build_luck_decades(fp: FourPillars, direction: int, start_age: float, count: int = 8) -> list[dict]:
    base = pillar_to_index(fp.month_gan, fp.month_zhi)
    decades = []
    age0 = int(start_age)
    for i in range(1, count + 1):
        idx = (base + direction * i) % 60
        gan, zhi = SEXAGENARY_60[idx]
        decades.append({"age": age0 + (i - 1) * 10, "gan": gan, "zhi": zhi})
    return decades

def calc_luck(local_dt: datetime, fp: FourPillars, sex: str = "M") -> dict:
    dir_ = luck_direction(fp.year_gan, sex)
    start_age, prev_bd, next_bd, policy = calc_luck_start_age(local_dt, dir_)
    decades = build_luck_decades(fp, dir_, start_age)
    return {
        "direction": "forward" if dir_ == +1 else "backward",
        "start_age": start_age,
        "boundary_policy": policy,
        "prev_boundary_local": prev_bd.isoformat(),
        "next_boundary_local": next_bd.isoformat(),
        "decades": decades,
    }

# -------------------------
# 11단계: 입력 달력(양/음) & 일주 경계 정책
# -------------------------
def to_local_dt_from_input(birth_local: str, tz: str, calendar: str = "solar", lunar_is_leap_month: bool = False) -> datetime:
    dt_naive = datetime.strptime(birth_local, "%Y-%m-%d %H:%M:%S")
    if calendar.lower() == "solar":
        return dt_naive.replace(tzinfo=ZoneInfo(tz))
    if not HAS_LUNARPY:
        raise RuntimeError("lunar-python이 없어 음력 입력을 처리할 수 없습니다.")
    from lunar_python import Lunar
    y, m, d = dt_naive.year, dt_naive.month, dt_naive.day
    hh, mm, ss = dt_naive.hour, dt_naive.minute, dt_naive.second
    try:
        lunar = Lunar.fromYmdHms(y, m, d, hh, mm, ss, lunar_is_leap_month)
    except Exception:
        lunar = Lunar.fromYmd(y, m, d)
    solar = lunar.getSolar()
    try:
        yy, mm_, dd = solar.getYear(), solar.getMonth(), solar.getDay()
        H = getattr(solar, "getHour", lambda: hh)()
        M = getattr(solar, "getMinute", lambda: mm)()
        S = getattr(solar, "getSecond", lambda: ss)()
    except Exception:
        yy, mm_, dd, H, M, S = y, m, d, hh, mm, ss
    return datetime(yy, mm_, dd, H, M, S, tzinfo=ZoneInfo(tz))

def calc_day_pillar_with_policy(local_dt: datetime, policy: str = "midnight"):
    dt = local_dt
    if policy == "zi23":
        if local_dt.hour >= 23:
            dt = local_dt + timedelta(days=1)
    elif policy == "zi-split":
        if local_dt.hour == 23:
            dt = local_dt + timedelta(days=1)
        elif local_dt.hour == 0:
            dt = local_dt - timedelta(days=1)
    return calc_day_pillar(dt)

# -------------------------
# 네 기둥 종합 (정밀 연/월 + 일주 경계 정책 지원)
# -------------------------
def build_four_pillars(
    local_dt: datetime,
    day_boundary_policy: str = "midnight",
    *,
    prefer_precise_day_hour: bool = True,
    use_true_solar_time: bool = False,
    longitude_deg: float | None = None,
) -> FourPillars:
    """
    - prefer_precise_day_hour=True: 가능하면 EightChar로 '일/시'까지 정밀 산출(권장)
    - use_true_solar_time=True: 경도 기반 LMT 보정(시주가 戌/亥 경계에서 달라지는 문제 대응)
    """
    # 1) lunar_python로 연/월/일/시 모두 뽑기 시도 (일/시 분리 계산)
    if prefer_precise_day_hour and HAS_LUNARPY:
        precise = calc_pillars_precise_all(local_dt, use_true_solar_time=use_true_solar_time, longitude_deg=longitude_deg)
        if precise:
            (yg, yz), (mg, mz), (dg, dz), (hg, hz), _dt_used = precise
            fp = FourPillars(yg, yz, mg, mz, dg, dz, hg, hz)
            fp._boundary_note = "precise_jieqi"
            fp._day_boundary_policy = day_boundary_policy
            fp._solar_time_mode = "apparent" if (use_true_solar_time and longitude_deg is not None) else "zone"
            fp._longitude_deg = longitude_deg
            return fp  # ← 정밀 경로 성공 시 바로 반환

    # 2) 폴백: 기존 방식 (연/월은 정밀 가능 시 사용, 일주는 정책 적용)
    day_gan, day_zhi, *_ = calc_day_pillar_with_policy(local_dt, day_boundary_policy)
    dt_for_hour = adjust_to_apparent_solar_time(local_dt, longitude_deg) if (use_true_solar_time and longitude_deg is not None) else local_dt
    hour_branch_han, _kor, hb_idx = calc_hour_branch(dt_for_hour)
    hour_gan_han = calc_hour_stem(day_gan, hb_idx)

    precise_year, precise_month = calc_year_month_pillars_precise(local_dt)
    if precise_year and precise_month:
        y_gan, y_zhi = precise_year
        m_gan, m_zhi = precise_month
        boundary_note = "precise_jieqi"
    else:
        y_gan, y_zhi = calc_year_pillar(local_dt)
        m_gan, m_zhi, _ = calc_month_pillar(local_dt, y_gan)
        boundary_note = "approx_jieqi"

    fp = FourPillars(
        year_gan=y_gan, year_zhi=y_zhi,
        month_gan=m_gan, month_zhi=m_zhi,
        day_gan=day_gan, day_zhi=day_zhi,
        hour_gan=hour_gan_han, hour_zhi=hour_branch_han
    )
    fp._boundary_note = boundary_note
    fp._day_boundary_policy = day_boundary_policy
    fp._solar_time_mode = "apparent" if (use_true_solar_time and longitude_deg is not None) else "zone"
    fp._longitude_deg = longitude_deg
    return fp

# -------------------------
# 전체 결과 JSON (오행/십성 포함)
# -------------------------
def compute_bazi_payload(
    birth_local: str,
    tz: str,
    sex: str = "F",
    *,
    calendar: str = "solar",
    lunar_is_leap_month: bool = False,
    day_boundary_policy: str = "midnight",
    # ↓↓↓ 추가 옵션 ↓↓↓
    prefer_precise_day_hour: bool = True,   # EightChar로 일/시까지 정밀 산출
    use_true_solar_time: bool = False,      # 진태양시(LMT) 보정 사용
    longitude_deg: float | None = None,     # LMT 보정 시 사용할 경도(예: 서울 126.9784)
    # 집계/표시 모드
    wuxing_mode: str = "with_hidden",       # "visible_only" | "with_hidden" | "with_branch" | "hidden_main_only"
    ten_gods_mode: str = "with_hidden",     # "visible_only" | "main_only" | "with_hidden"
    hidden_order: str = "std",              # "std" | "cmj" | "mjc"
) -> dict:
    # 1) 입력을 로컬 datetime으로 (양력/음력 모두 지원)
    dt_local = to_local_dt_from_input(
        birth_local, tz,
        calendar=calendar,
        lunar_is_leap_month=lunar_is_leap_month
    )

    # 2) (표시/계산용) ISO, JD 값을 '실제 계산에 사용된 dt_local' 기준으로 생성
    dt_utc = to_utc(dt_local)
    jdn, jd = gregorian_to_jd(dt_utc)
    local_iso = dt_local.isoformat()
    utc_iso = dt_utc.isoformat().replace("+00:00", "Z")

    # 3) 네 기둥 — 정밀(일/시) + LMT 옵션 반영
    fp = build_four_pillars(
        dt_local,
        day_boundary_policy=day_boundary_policy,
        prefer_precise_day_hour=prefer_precise_day_hour,
        use_true_solar_time=use_true_solar_time,
        longitude_deg=longitude_deg,
    )

    # 4) 지장간 / 오행 / 십성 / 대운
    hidden = compute_hidden_stems(fp, order=hidden_order)
    wuxing_detail = compute_wuxing_distribution(fp, mode=wuxing_mode, hidden_order=hidden_order)
    ten_visible = compute_ten_gods_visible(fp)
    ten_counts  = compute_ten_gods_counts(fp, mode=ten_gods_mode)
    luck = calc_luck(dt_local, fp, sex=sex)

    # --- (추가) 보기 전용 라벨 블록 만들기 ---
    pillars_label = {
        "year":  {"gan": label_gan(fp.year_gan),  "zhi": label_zhi(fp.year_zhi)},
        "month": {"gan": label_gan(fp.month_gan), "zhi": label_zhi(fp.month_zhi)},
        "day":   {"gan": label_gan(fp.day_gan),   "zhi": label_zhi(fp.day_zhi)},
        "hour":  {"gan": label_gan(fp.hour_gan),  "zhi": label_zhi(fp.hour_zhi)},
        "combined": {
            "year":  f"{label_gan(fp.year_gan)}{label_zhi(fp.year_zhi)}",
            "month": f"{label_gan(fp.month_gan)}{label_zhi(fp.month_zhi)}",
            "day":   f"{label_gan(fp.day_gan)}{label_zhi(fp.day_zhi)}",
            "hour":  f"{label_gan(fp.hour_gan)}{label_zhi(fp.hour_zhi)}",
        }
    }

    hidden_stems_label = {
        k: [label_gan(s) for s in v] for k, v in hidden.items()
    }

    wuxing_label = {
        "visible_only": { WUXING_KEY_LABEL[k]: v for k, v in wuxing_detail["visible_only"].items() },
        "with_hidden":  { WUXING_KEY_LABEL[k]: v for k, v in wuxing_detail["with_hidden"].items()  },
        "branch_only":  { WUXING_KEY_LABEL[k]: v for k, v in wuxing_detail["branch_only"].items()  },
        "hidden_main_only": { WUXING_KEY_LABEL[k]: v for k, v in wuxing_detail["hidden_main_only"].items()  },
        "selected_counts": { WUXING_KEY_LABEL[k]: v for k, v in wuxing_detail["selected_counts"].items() },
        "mode": wuxing_detail["mode"],
    }

    luck_decades_label = [
        {"age": d["age"], "gan": label_gan(d["gan"]), "zhi": label_zhi(d["zhi"])}
        for d in luck.get("decades", [])
    ]

    # 5) 출력 JSON
    payload = {
        "input": {
            "birth_local": birth_local,
            "tz": tz,
            "sex": sex.upper(),
            "calendar": calendar,
            "lunar_is_leap_month": lunar_is_leap_month,
            "local_iso": local_iso,
            "utc_iso": utc_iso,
            "jdn": jdn,
            "jd": round(jd, 6),
        },
        "pillars": {
            "year":  {"gan": fp.year_gan,  "zhi": fp.year_zhi},
            "month": {"gan": fp.month_gan, "zhi": fp.month_zhi},
            "day":   {"gan": fp.day_gan,   "zhi": fp.day_zhi},
            "hour":  {"gan": fp.hour_gan,  "zhi": fp.hour_zhi},
            "boundary_policy": getattr(fp, "_boundary_note", "approx_jieqi"),
            "day_boundary_policy": getattr(fp, "_day_boundary_policy", "midnight"),
            "solar_time_mode": getattr(fp, "_solar_time_mode", "zone"),
            "longitude_deg": getattr(fp, "_longitude_deg", None),
        },
        "hidden_stems": hidden,
        "wuxing": {
            "mode": wuxing_detail["mode"],
            "visible_only": wuxing_detail["visible_only"],
            "with_hidden":  wuxing_detail["with_hidden"],
            "branch_only":  wuxing_detail["branch_only"],
            "hidden_main_only": wuxing_detail["hidden_main_only"],
            "selected_counts": wuxing_detail["selected_counts"],
        },
        "ten_gods": {
            "mode": ten_gods_mode,
            "visible_by_pillar": ten_visible,
            "counts_with_hidden": compute_ten_gods_counts(fp, mode="with_hidden"),
            "selected_counts": ten_counts,  # ← 화면/요약에 사용할 값
        },
        "luck": luck,
        # ---- (추가) 보기 전용 라벨들 ----
        "pillars_label": pillars_label,
        "hidden_stems_label": hidden_stems_label,
        "wuxing_label": wuxing_label,
        "luck_label": {"decades": luck_decades_label},
        "meta": {
            "version": "0.14-selectable-modes",
            "notes": "adds selectable wuxing/ten_gods modes, hidden order, and selected_counts",
        },
    }
    return payload

# -------------------------
# 12단계: 요약 문장 생성기
# -------------------------
def _safe_get(d, *path, default=None):
    cur = d
    try:
        for key in path:
            cur = cur[key]
        return cur
    except Exception:
        return default

def _sort_desc_counts(counts: dict):
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))

def _wuxing_kor_name(en):
    return {"wood":"목","fire":"화","earth":"토","metal":"금","water":"수"}.get(en, en)

def summarize_wuxing(payload: dict) -> str:
    # 요약은 항상 'selected_counts'를 사용 (UI/모드와 일치)
    counts = _safe_get(payload, "wuxing", "selected_counts", default={})
    if not counts:
        # 폴백: 구버전 필드
        counts = _safe_get(payload, "wuxing", "with_hidden", default={})
    if not counts:
        return "오행 분포 정보를 계산할 수 없습니다."
    parts = [f"{_wuxing_kor_name(k)} {v}" for k,v in _sort_desc_counts(counts)]
    strongest = max(counts, key=counts.get)
    weakest   = min(counts, key=counts.get)
    return f"오행 분포: {' / '.join(parts)}. 강점은 {_wuxing_kor_name(strongest)}, 약점은 {_wuxing_kor_name(weakest)} 쪽입니다."

TEN_GOD_BRIEF = {
    "비견":"자존·동료·독립성",
    "겁재":"경쟁·소모·의리",
    "식신":"생산·표현·건강",
    "상관":"창의·도전·반골",
    "편재":"유연한 돈·활동성",
    "정재":"안정 수입·실리",
    "편관":"도전·압박·경쟁",
    "정관":"규범·책임·직장",
    "편인":"직감·아이디어",
    "정인":"학습·자격·보호",
}

def summarize_ten_gods(payload: dict) -> str:
    # 요약은 항상 'selected_counts'를 사용
    counts = _safe_get(payload, "ten_gods", "selected_counts", default={})
    if not counts:
        counts = _safe_get(payload, "ten_gods", "counts_with_hidden", default={})
    if not counts:
        return "십성 분포를 계산할 수 없습니다."
    ordered = _sort_desc_counts(counts)
    top = [tg for tg,c in ordered if c>0][:3]
    bullet = " / ".join([f"{tg}({counts[tg]})" for tg in top])
    gloss  = " · ".join([TEN_GOD_BRIEF.get(tg, tg) for tg in top])
    return f"십성 상위: {bullet}. 키워드: {gloss}."

def summarize_ten_visible(payload: dict) -> str:
    vis = _safe_get(payload, "ten_gods", "visible_by_pillar", default={})
    if not vis:
        return "겉간 기준 십성 정보가 없습니다."
    mapping = {"year":"연간","month":"월간","day":"일간","hour":"시간"}
    pairs = [f"{mapping[k]}={vis[k]}" for k in ["year","month","day","hour"] if vis.get(k)]
    return "겉간 십성: " + ", ".join(pairs) + "."

def summarize_luck(payload: dict) -> str:
    luck = payload.get("luck", {})
    direction = luck.get("direction","")
    start_age = luck.get("start_age", "")
    dir_kor = "순행" if direction=="forward" else "역행"
    decades = luck.get("decades", [])[:4]
    lst = ", ".join([f"{d['age']}세 {d['gan']}{d['zhi']}" for d in decades]) if decades else "정보 없음"
    return f"대운은 {dir_kor}이며 시작나이는 약 {start_age}세입니다. 주요 대운: {lst} …"

def summarize_pillars(payload: dict) -> str:
    p = payload.get("pillars", {})
    y = p.get("year", {}); m = p.get("month", {}); d = p.get("day", {}); h = p.get("hour", {})
    note = p.get("boundary_policy","")
    return (f"네 기둥: 연 {y.get('gan','?')}{y.get('zhi','?')}, "
            f"월 {m.get('gan','?')}{m.get('zhi','?')}, "
            f"일 {d.get('gan','?')}{d.get('zhi','?')}, "
            f"시 {h.get('gan','?')}{h.get('zhi','?')} "
            f"(경계정책: {note}).")

def generate_text_report(payload: dict) -> str:
    parts = []
    parts.append(summarize_pillars(payload))
    parts.append(summarize_wuxing(payload))
    parts.append(summarize_ten_gods(payload))
    parts.append(summarize_ten_visible(payload))
    parts.append(summarize_luck(payload))
    parts.append("※ 본 해석은 전통 명리의 일반 규칙을 바탕으로 한 안내입니다. 중요한 의사결정의 단일 근거로 삼기보다 참고자료로 활용하세요.")
    return "\n".join(parts)

# -------------------------
# 실행 데모
# -------------------------
if __name__ == "__main__":
    demo = BirthInput(birth_local="1995-08-26 16:00:00", tz="Asia/Seoul")
    dt_local = parse_local_dt(demo.birth_local, demo.tz)

    # 네 기둥
    fp = build_four_pillars(dt_local)
    print("[Four Pillars]")
    print(f" Year : {fp.year_gan}{fp.year_zhi}")
    print(f" Month: {fp.month_gan}{fp.month_zhi}")
    print(f" Day  : {fp.day_gan}{fp.day_zhi}")
    print(f" Hour : {fp.hour_gan}{fp.hour_zhi}")

    # 지장간
    hs = compute_hidden_stems(fp, order="cmj")
    print("\n[Hidden Stems — order=cmj]")
    for k,v in hs.items():
        print(f" {k:5s}: {' '.join(v) if v else '-'}")

    # 오행 분포 (선택 모드 테스트)
    dist = compute_wuxing_distribution(fp, mode="with_branch", hidden_order="cmj")
    print("\n[Wuxing Counts]")
    print(" visible_only      :", dist["visible_only"])
    print(" with_hidden       :", dist["with_hidden"])
    print(" branch_only       :", dist["branch_only"])
    print(" hidden_main_only  :", dist["hidden_main_only"])
    print(" selected (mode=with_branch):", dist["selected_counts"])

    # 십성 출력
    tg_vis = compute_ten_gods_visible(fp)
    print("\n[Ten Gods — visible]")
    for k in ["year","month","day","hour"]:
        print(f" {k:5s}: {tg_vis[k]}")
    tg_counts = compute_ten_gods_counts(fp, mode="main_only")
    print("\n[Ten Gods — counts (mode=main_only)]")
    print(tg_counts)

    # 대운
    sex = "F"
    luck = calc_luck(dt_local, fp, sex=sex)
    print("\n[DaYun]")
    print(" direction :", luck["direction"])
    print(" start_age :", luck["start_age"])
    print(" boundary  :", "prev =", luck["prev_boundary_local"], "| next =", luck["next_boundary_local"])
    print(" decades   :")
    for d in luck["decades"]:
        print(f"  - age {d['age']:>2}: {d['gan']}{d['zhi']}")

    # 전체 JSON
    sample = compute_bazi_payload("1995-08-26 16:00:00", "Asia/Seoul", sex="F",
                                  wuxing_mode="with_branch", ten_gods_mode="main_only", hidden_order="cmj")
    print("\n[JSON payload]")
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    # 요약 문장
    report = generate_text_report(sample)
    print("\n[Text Summary]")
    print(report)
