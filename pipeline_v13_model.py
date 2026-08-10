"""한우 최종 등급 예측 모델.

팀명: 부모님이 누구니
참가자: 김은호, 정재용

본 모델은 한우 최종 등급을 직접 예측하는 경로와, 실제 등급 산정 구조를 반영해
육질등급과 육량등급을 분리 예측하는 계층형 경로를 결합한다. LightGBM은 최종
16개 등급의 전체 결정 경계를 학습하고, CatBoost는 KPN·혈통·지역 등 범주형
신호가 중요한 육질등급을 예측하며, XGBoost는 체중·도체형질 proxy와 순서형
경계가 중요한 육량등급을 담당한다.

핵심 개선점은 육량 C등급 재현율 보완이다. 일반적인 A/B/C 다중분류에서는 표본
수가 많은 B등급이 우세해 실제 C등급 개체가 B로 흡수되는 문제가 있었다. 이를
해결하기 위해 육량등급을 먼저 C/non-C로 분해하고, non-C 내부에서 A/B를 다시
구분하는 조건부 확률 구조를 적용하였다. C를 적극적으로 탐지하는 모델과 B/C
경계를 보수적으로 유지하는 모델을 함께 사용해 C recall 개선과 B등급 성능
보존의 균형을 맞췄다.

추가 혈통 및 KPN 데이터는 단순한 과거 등급 암기가 아니라 유전적 경향을 반영하는
보조 신호로 활용하였다. 또한 테스트 데이터에 없는 도체형질은 실제 값을 직접
사용하지 않고, 농장 단위 누수를 방지하는 nested proxy 모델로 복원한 예측값만
사용하여 학습 조건과 추론 조건을 일치시켰다. 최종 보정과 앙상블 가중치 선택은
교차 적합 방식으로 수행해 검증 Fold를 직접 보고 조정하는 낙관적 평가를 줄였다.
"""

import ast
import gc
import os
import re
import time
import warnings
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_fscore_support, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# 권장 Windows 실행 환경:
#   conda create -n hanwoo python=3.11 -y
#   conda activate hanwoo
#   python -m pip install --upgrade pip
#   python -m pip install numpy pandas scikit-learn lightgbm catboost xgboost openpyxl
#
# PowerShell 전체 GPU 실행:
#   cd path\to\livestock-quality-prediction
#   conda activate hanwoo
#   $env:CATBOOST_TASK_TYPE = "GPU"
#   $env:XGBOOST_DEVICE = "cuda"
#   $env:PIPELINE_SMOKE_STAGE = ""
#   $env:SAVE_OOF = "1"
#   python .\pipeline_v13_model.py
#
# The competition runs were developed with GPU acceleration. The public
# release defaults to CPU so smoke checks and first runs are portable.
#
# GPU/CUDA 문제가 있을 때 CPU 실행:
#   $env:CATBOOST_TASK_TYPE = "CPU"
#   $env:XGBOOST_DEVICE = "cpu"
# 권장 실행:
#   CatBoost/XGBoost GPU 가능 환경: CATBOOST_TASK_TYPE=GPU, XGBOOST_DEVICE=cuda
#   Mac/CPU 환경: CATBOOST_TASK_TYPE=CPU, XGBOOST_DEVICE=cpu


# -----------------------------------------------------------------------------
# Data processing
# -----------------------------------------------------------------------------

def _decode_grade(value):
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    text = str(value).strip()
    if text.startswith("b'") or text.startswith('b"'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, bytes):
                return parsed.decode("utf-8", errors="ignore")
        except Exception:
            return text
    return text


def _column_index(cell_ref):
    index = 0
    for char in cell_ref:
        if char.isalpha():
            index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index


def _read_xlsx_rows(path):
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    }
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for shared_item in shared_root.findall("main:si", namespace):
                text_parts = [node.text or "" for node in shared_item.iterfind(".//main:t", namespace)]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//main:sheetData/main:row", namespace):
            values = []
            next_col = 1
            for cell in row.findall("main:c", namespace):
                ref = cell.attrib.get("r", "")
                col_index = _column_index(ref)
                while next_col < col_index:
                    values.append(None)
                    next_col += 1

                cell_type = cell.attrib.get("t")
                cell_value = None
                if cell_type == "s":
                    value_node = cell.find("main:v", namespace)
                    if value_node is not None and value_node.text is not None:
                        cell_value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr":
                    cell_value = "".join(node.text or "" for node in cell.iterfind(".//main:t", namespace))
                else:
                    value_node = cell.find("main:v", namespace)
                    if value_node is not None:
                        cell_value = value_node.text

                values.append(cell_value)
                next_col = col_index + 1
            rows.append(values)

        width = max((len(row) for row in rows), default=0)
        normalized = [row + [None] * (width - len(row)) for row in rows]
        return normalized


class HanwooDataProcessorV8:
    QUALITY_ORDER = ["1++", "1+", "1", "2", "3", "등외"]
    YIELD_ORDER = ["A", "B", "C"]
    FINAL_GRADES = [
        "1++A",
        "1++B",
        "1++C",
        "1+A",
        "1+B",
        "1+C",
        "1A",
        "1B",
        "1C",
        "2A",
        "2B",
        "2C",
        "3A",
        "3B",
        "3C",
        "등외",
    ]
    GRADE_SCORE = {grade: 15 - index for index, grade in enumerate(FINAL_GRADES)}
    CATEGORICAL_COLUMNS = [
        "KPN_NO",
        "stn",
        "sido",
        "sigungu",
        "sex_code",
        "birth_month",
        "abatt_month",
        "birth_season",
        "abatt_season",
        "farm_size",
    ]

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.kpn_bv = None
        self.lineage = None
        self.weather_monthly = None
        self.weather_daily = None
        self.farm_profile = None
        self.kpn_stats = None
        self.stn_stats = None

    def _find_first(self, columns, keywords):
        for column in columns:
            text = str(column)
            if all(keyword in text for keyword in keywords):
                return column
        return None

    def _build_kpn_features(self, kpn_df):
        columns = list(kpn_df.columns)
        key_col = self._find_first(columns, ["KPN"])
        if key_col is None:
            return None

        kpn_df[key_col] = kpn_df[key_col].astype(str).str.strip()
        feature_specs = [
            (["12개월체중", "육종가"], "kpn_weight_bv"),
            (["12개월체중", "정확도"], "kpn_weight_accuracy"),
            (["도체중", "육종가"], "kpn_carcass_weight_bv"),
            (["도체중", "정확도"], "kpn_carcass_weight_accuracy"),
            (["등심단면적", "육종가"], "kpn_rea_bv"),
            (["등심단면적", "정확도"], "kpn_rea_accuracy"),
            (["등지방두께", "육종가"], "kpn_backfat_bv"),
            (["등지방두께", "정확도"], "kpn_backfat_accuracy"),
            (["근내지방도", "육종가"], "kpn_insfat_bv"),
            (["근내지방도", "정확도"], "kpn_insfat_accuracy"),
        ]
        feature_map = {
            source: target
            for keywords, target in feature_specs
            if (source := self._find_first(columns, keywords)) is not None
        }
        feature_cols = [key_col] + [source for source in feature_map if source is not None]
        if len(feature_cols) == 1:
            return None

        rename_map = {key_col: "KPN_NO"}
        rename_map.update({source: target for source, target in feature_map.items() if source is not None})
        feature_frame = kpn_df[feature_cols].rename(columns=rename_map)
        for column in feature_frame.columns:
            if column != "KPN_NO":
                feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
        return feature_frame.drop_duplicates("KPN_NO")

    def _build_season(self, month_series):
        month = month_series.fillna(0).astype(int)
        season = pd.Series(-1, index=month.index, dtype="int64")
        season.loc[month.isin([3, 4, 5])] = 0
        season.loc[month.isin([6, 7, 8])] = 1
        season.loc[month.isin([9, 10, 11])] = 2
        season.loc[month.isin([12, 1, 2])] = 3
        return season

    def load_auxiliary_data(self):
        print("[DataProcessorV8] Loading auxiliary data...")

        kpn_path = os.path.join(self.data_dir, "KPN 유전능력 자료.xlsx")
        if os.path.exists(kpn_path):
            try:
                rows = _read_xlsx_rows(kpn_path)
                if rows:
                    kpn_df = pd.DataFrame(rows[1:], columns=rows[0])
                else:
                    kpn_df = pd.DataFrame()
                self.kpn_bv = self._build_kpn_features(kpn_df)
            except Exception as exc:
                print(f"[DataProcessorV8] Skipping KPN Excel file: {exc}")
                self.kpn_bv = None

        lineage_path = os.path.join(self.data_dir, "hanwoo_lineage_0612.csv")
        if not os.path.exists(lineage_path):
            lineage_path = os.path.join(self.data_dir, "hanwoo_lineage.csv")
        if os.path.exists(lineage_path):
            self.lineage = pd.read_csv(lineage_path, usecols=["CATTLE_NO", "KPN_NO"])
            self.lineage["CATTLE_NO"] = self.lineage["CATTLE_NO"].astype(str).str.strip()
            self.lineage["KPN_NO"] = self.lineage["KPN_NO"].astype(str).str.strip()
            self.lineage = self.lineage.drop_duplicates("CATTLE_NO")

        weather_path = os.path.join(self.data_dir, "hanwoo_weather.csv")
        if os.path.exists(weather_path):
            weather = pd.read_csv(weather_path, parse_dates=["date"])
            weather["ta_avg"] = (weather["ta_max"] + weather["ta_min"]) / 2
            weather["thi"] = 0.8 * weather["ta_avg"] + (weather["rhm_avg"] / 100) * (weather["ta_avg"] - 14.3) + 46.4
            weather["is_heat_day"] = (weather["ta_max"] >= 33).astype(int)
            self.weather_daily = weather

            weather["year"] = weather["date"].dt.year
            weather["month"] = weather["date"].dt.month
            wm = weather.groupby(["stn", "year", "month"]).agg(
                s_ta=("ta_avg", "mean"),
                s_rn=("rn_day", "sum"),
                s_rhm=("rhm_avg", "mean"),
                s_thi=("thi", "mean"),
                s_heat_days=("is_heat_day", "sum"),
            ).reset_index()

            wm = wm.sort_values(["stn", "year", "month"])
            weather_cols = ["s_ta", "s_rn", "s_rhm", "s_thi", "s_heat_days"]
            wm[weather_cols] = wm.groupby("stn")[weather_cols].shift(1)
            for window in [3, 6, 12]:
                wm[f"s_ta_{window}m"] = wm.groupby("stn")["s_ta"].transform(
                    lambda series: series.rolling(window, min_periods=1).mean()
                )
                wm[f"s_thi_{window}m"] = wm.groupby("stn")["s_thi"].transform(
                    lambda series: series.rolling(window, min_periods=1).mean()
                )
                wm[f"s_heat_{window}m"] = wm.groupby("stn")["s_heat_days"].transform(
                    lambda series: series.rolling(window, min_periods=1).sum()
                )

            wm["s_ta_trend"] = wm["s_ta_3m"] - wm["s_ta_12m"]
            wm["s_thi_trend"] = wm["s_thi_3m"] - wm["s_thi_12m"]
            wm["s_heat_rate_3m"] = wm["s_heat_3m"] / 3.0
            wm["s_heat_rate_6m"] = wm["s_heat_6m"] / 6.0
            wm["s_heat_rate_12m"] = wm["s_heat_12m"] / 12.0

            self.weather_monthly = wm.astype({"stn": int, "year": int, "month": int}).astype("float32", errors="ignore")
            self.weather_monthly[["stn", "year", "month"]] = self.weather_monthly[["stn", "year", "month"]].astype(int)

        area_path = os.path.join(self.data_dir, "hanwoo_area.csv")
        death_path = os.path.join(self.data_dir, "hanwoo_death.csv")
        if os.path.exists(area_path):
            area = pd.read_csv(area_path)
            year_cols = [col for col in area.columns if re.fullmatch(r"C\d{4}", str(col))]
            area = area.drop_duplicates("FARM_UNIQUE_NO", keep="last")

            if os.path.exists(death_path):
                death = pd.read_csv(death_path)
                death_cnt = death.groupby("FARM_UNIQUE_NO").size().reset_index(name="death_cnt")
                area = area.merge(death_cnt, on="FARM_UNIQUE_NO", how="left")
            else:
                area["death_cnt"] = 0

            area["death_cnt"] = area["death_cnt"].fillna(0)
            if year_cols:
                area = area.melt(
                    id_vars=["FARM_UNIQUE_NO", "AREA", "death_cnt"],
                    value_vars=year_cols,
                    var_name="profile_year",
                    value_name="livestock_count",
                )
                area["profile_year"] = area["profile_year"].str[1:].astype(int)
            else:
                area["profile_year"] = -1
                area["livestock_count"] = np.nan

            area_count = pd.to_numeric(area["livestock_count"], errors="coerce").fillna(0)
            area_size = pd.to_numeric(area["AREA"], errors="coerce").fillna(0)
            area["density"] = (area_size / (area_count + 1)).astype("float32")
            farm_size = pd.cut(
                area_count,
                bins=[-1, 30, 100, np.inf],
                labels=[0, 1, 2],
            )
            area["farm_size"] = farm_size.cat.add_categories([-1]).fillna(-1).astype("int64")
            self.farm_profile = area[
                ["FARM_UNIQUE_NO", "profile_year", "density", "farm_size", "death_cnt"]
            ].copy()
            self.farm_profile["FARM_UNIQUE_NO"] = self.farm_profile["FARM_UNIQUE_NO"].astype(str).str.strip()

    def fit_target_stats(self, train_df):
        print("[DataProcessorV8] Calculating base target statistics...")
        df = train_df.copy()
        df["LAST_GRADE"] = df["LAST_GRADE"].map(_decode_grade).fillna("등외").astype(str)
        df["grade_score"] = df["LAST_GRADE"].map(self.GRADE_SCORE)

        if "KPN_NO" not in df.columns and self.lineage is not None:
            df["CATTLE_NO"] = df["CATTLE_NO"].astype(str).str.strip()
            df = df.merge(self.lineage, on="CATTLE_NO", how="left")

        if "KPN_NO" in df.columns:
            df["KPN_NO"] = df["KPN_NO"].astype(str).str.strip()
            self.kpn_stats = (
                df.groupby("KPN_NO")["grade_score"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "kpn_grade_avg", "count": "kpn_grade_cnt"})
                .reset_index()
            )

        if "stn" in df.columns:
            self.stn_stats = (
                df.groupby("stn")["grade_score"]
                .mean()
                .reset_index()
                .rename(columns={"grade_score": "stn_grade_avg"})
            )

    def transform(self, df, is_train=True):
        print(f"[DataProcessorV8] Transforming {'train' if is_train else 'test'} data...")
        df = df.copy()

        for col in ["CATTLE_NO", "FARM_UNIQUE_NO", "KPN_NO"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        df["ABATT_DATE"] = pd.to_datetime(df["ABATT_DATE"], errors="coerce")
        df["BIRTH_YMD"] = pd.to_datetime(df["BIRTH_YMD"].astype(str), format="%Y%m%d", errors="coerce")
        df["abatt_year"] = df["ABATT_DATE"].dt.year.fillna(0).astype(int)
        df["abatt_month"] = df["ABATT_DATE"].dt.month.fillna(0).astype(int)
        df["birth_year"] = df["BIRTH_YMD"].dt.year.fillna(0).astype(int)
        df["birth_month"] = df["BIRTH_YMD"].dt.month.fillna(0).astype(int)

        df["abatt_season"] = self._build_season(df["abatt_month"])
        df["birth_season"] = self._build_season(df["birth_month"])
        df["rearing_months"] = (
            df["abatt_year"] * 12 + df["abatt_month"] - (df["birth_year"] * 12 + df["birth_month"])
        ).clip(lower=0)
        df["age_sq"] = (df["AGE"] ** 2).astype("float32")
        df["age_log"] = np.log1p(df["AGE"].clip(lower=0)).astype("float32")
        df["weight_log"] = np.log1p(df["WEIGHT"].clip(lower=0)).astype("float32")
        df["weight_per_age"] = (df["WEIGHT"] / (df["AGE"] + 1)).astype("float32")
        df["weight_x_age"] = (df["WEIGHT"] * df["AGE"]).astype("float32")
        df["abatt_month_sin"] = np.sin(2 * np.pi * df["abatt_month"] / 12.0).astype("float32")
        df["abatt_month_cos"] = np.cos(2 * np.pi * df["abatt_month"] / 12.0).astype("float32")
        df["birth_month_sin"] = np.sin(2 * np.pi * df["birth_month"] / 12.0).astype("float32")
        df["birth_month_cos"] = np.cos(2 * np.pi * df["birth_month"] / 12.0).astype("float32")

        sex_map = {"암": 0, "수": 1, "거세": 2}
        if "JUDGE_SEX" in df.columns:
            df["sex_code"] = df["JUDGE_SEX"].map(sex_map).fillna(-1).astype(int)
        else:
            df["sex_code"] = -1

        if self.weather_monthly is not None and "stn" in df.columns:
            df = df.merge(
                self.weather_monthly,
                left_on=["stn", "abatt_year", "abatt_month"],
                right_on=["stn", "year", "month"],
                how="left",
            )
            df.drop(columns=["year", "month"], inplace=True)

        if self.farm_profile is not None and "FARM_UNIQUE_NO" in df.columns:
            df = df.merge(
                self.farm_profile,
                left_on=["FARM_UNIQUE_NO", "abatt_year"],
                right_on=["FARM_UNIQUE_NO", "profile_year"],
                how="left",
            )
            df.drop(columns=["profile_year"], inplace=True)

        if self.lineage is not None and "CATTLE_NO" in df.columns:
            df = df.merge(self.lineage, on="CATTLE_NO", how="left")

        if self.kpn_bv is not None and "KPN_NO" in df.columns:
            df = df.merge(self.kpn_bv, on="KPN_NO", how="left")

        if self.kpn_stats is not None and "KPN_NO" in df.columns:
            df = df.merge(self.kpn_stats, on="KPN_NO", how="left")
        if self.stn_stats is not None and "stn" in df.columns:
            df = df.merge(self.stn_stats, on="stn", how="left")

        if is_train and "LAST_GRADE" in df.columns:
            df["LAST_GRADE"] = df["LAST_GRADE"].map(_decode_grade).fillna("등외").astype(str)
            df["target_q"] = df["LAST_GRADE"].str.extract(r"^(1\+\+|1\+|1|2|3|등외)")[0].fillna("등외")
            df["target_y"] = df["LAST_GRADE"].str.extract(r"(A|B|C)$")[0].fillna("B")

        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("string").fillna("__MISSING__").astype("category")

        drop_cols = ["JUDGE_SEX", "CATTLE_NO", "ABATT_DATE", "BIRTH_YMD", "JUDGE_DATE", "eupmyeondong"]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        return df


class HanwooDataProcessorV12(HanwooDataProcessorV8):
    LINEAGE_COLUMNS = [
        "KPN_NO",
        "FATHER_CATTLE_NO",
        "MOTHER_ANIMAL_NO",
        "F_GMOTHER_ANIMAL_NO",
        "F_GFATHER_CATTLE_NO",
        "M_GMOTHER_ANIMAL_NO",
        "M_GFATHER_CATTLE_NO",
    ]
    RAW_CATEGORICAL_LINEAGE = [
        "FATHER_CATTLE_NO",
        "F_GMOTHER_ANIMAL_NO",
        "F_GFATHER_CATTLE_NO",
        "M_GFATHER_CATTLE_NO",
    ]
    CATEGORICAL_COLUMNS = list(dict.fromkeys(
        HanwooDataProcessorV8.CATEGORICAL_COLUMNS + RAW_CATEGORICAL_LINEAGE
    ))

    def load_auxiliary_data(self):
        super().load_auxiliary_data()
        lineage_path = os.path.join(self.data_dir, "hanwoo_lineage_0612.csv")
        if not os.path.exists(lineage_path):
            lineage_path = os.path.join(self.data_dir, "hanwoo_lineage.csv")
        if os.path.exists(lineage_path):
            usecols = ["CATTLE_NO"] + self.LINEAGE_COLUMNS
            lineage = pd.read_csv(lineage_path, usecols=usecols)
            for column in usecols:
                lineage[column] = lineage[column].astype("string").str.strip()
            self.lineage = lineage.drop_duplicates("CATTLE_NO")


# -----------------------------------------------------------------------------
# Nested v13 proxy features
# -----------------------------------------------------------------------------

TRAIT_COLUMNS = ["BACKFAT", "REA", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]
PROXY_FEATURE_NAMES = [
    *(f"proxy_{column.lower()}" for column in TRAIT_COLUMNS),
    "proxy_windex",
    "proxy_yield_margin_a",
    "proxy_yield_margin_c",
    "proxy_yield_grade_signal",
    "proxy_insfat_rank",
    "proxy_yuksak_rank",
    "proxy_fatsak_rank",
    "proxy_tissue_rank",
    "proxy_quality_worst_rank",
    "proxy_growth_high",
]
BOUNDARY_TARGET_NAMES = [
    "boundary_windex",
    "boundary_yield_margin_a",
    "boundary_yield_margin_c",
    "boundary_insfat_rank",
    "boundary_yuksak_rank",
    "boundary_fatsak_rank",
    "boundary_tissue_rank",
    "boundary_quality_worst_rank",
    "boundary_growth_high",
]
BOUNDARY_PROXY_FEATURE_NAMES = [f"proxy_{name}" for name in BOUNDARY_TARGET_NAMES]


def _quality_ranks(predictions):
    insfat, yuksak, fatsak, tissue = [predictions[:, index] for index in [2, 3, 4, 5]]

    insfat_rank = np.select(
        [insfat >= 7.0, insfat >= 5.0, insfat >= 4.0, insfat >= 2.0],
        [0, 1, 2, 3],
        default=4,
    )
    rounded_yuksak = np.clip(np.rint(yuksak), 1, 7)
    yuksak_rank = np.select(
        [
            (rounded_yuksak >= 3) & (rounded_yuksak <= 5),
            np.isin(rounded_yuksak, [2, 6]),
            rounded_yuksak == 1,
            rounded_yuksak == 7,
        ],
        [0, 1, 2, 3],
        default=4,
    )
    rounded_fatsak = np.clip(np.rint(fatsak), 1, 7)
    fatsak_rank = np.select(
        [rounded_fatsak <= 4, rounded_fatsak == 5, rounded_fatsak == 6, rounded_fatsak == 7],
        [0, 1, 2, 3],
        default=4,
    )
    tissue_rank = np.clip(np.rint(tissue), 1, 5).astype("int8") - 1
    return [array.astype("float32") for array in [insfat_rank, yuksak_rank, fatsak_rank, tissue_rank]]


def derive_proxy_features(predictions, base_frame):
    """Create physical and rule-distance features from reconstructed traits."""
    predictions = np.asarray(predictions, dtype="float32").copy()
    clips = [(0, 60), (20, 200), (1, 9), (1, 7), (1, 7), (1, 5), (1, 9)]
    for index, (lower, upper) in enumerate(clips):
        predictions[:, index] = np.clip(predictions[:, index], lower, upper)

    result = pd.DataFrame(index=base_frame.index)
    for index, column in enumerate(TRAIT_COLUMNS):
        result[f"proxy_{column.lower()}"] = predictions[:, index]

    weight = pd.to_numeric(base_frame["WEIGHT"], errors="coerce").fillna(0).to_numpy(dtype="float32")
    sex = pd.to_numeric(base_frame["xgb_sex_code"], errors="coerce").fillna(2).to_numpy(dtype="int8")
    backfat, rea = predictions[:, 0], predictions[:, 1]
    intercept = np.select([sex == 0, sex == 1], [6.90137, 0.20108], default=11.06338)
    backfat_coef = np.select([sex == 0, sex == 1], [-0.9446, -2.18625], default=-1.25149)
    rea_coef = np.select([sex == 0, sex == 1], [0.31806, 0.29275], default=0.28238)
    weight_coef = np.select([sex == 0, sex == 1], [0.54952, 0.64099], default=0.56781)
    windex = (
        (intercept + backfat_coef * backfat + rea_coef * rea + weight_coef * weight)
        / np.maximum(weight, 1.0)
        * 100.0
    ).astype("float32")
    a_threshold = np.select([sex == 0, sex == 1], [61.83, 68.45], default=62.52)
    c_threshold = np.select([sex == 0, sex == 1], [59.70, 66.32], default=60.40)
    result["proxy_windex"] = windex
    result["proxy_yield_margin_a"] = (windex - a_threshold).astype("float32")
    result["proxy_yield_margin_c"] = (windex - c_threshold).astype("float32")
    result["proxy_yield_grade_signal"] = np.select(
        [windex >= a_threshold, windex >= c_threshold], [0, 1], default=2
    ).astype("float32")

    quality_ranks = _quality_ranks(predictions)
    for name, values in zip(
        ["proxy_insfat_rank", "proxy_yuksak_rank", "proxy_fatsak_rank", "proxy_tissue_rank"],
        quality_ranks,
    ):
        result[name] = values
    result["proxy_quality_worst_rank"] = np.maximum.reduce(quality_ranks).astype("float32")
    result["proxy_growth_high"] = (predictions[:, 6] >= 8.0).astype("float32")
    return result


def _yield_thresholds(base_frame):
    sex = pd.to_numeric(base_frame["xgb_sex_code"], errors="coerce").fillna(2).to_numpy(dtype="int8")
    a_threshold = np.select([sex == 0, sex == 1], [61.83, 68.45], default=62.52)
    c_threshold = np.select([sex == 0, sex == 1], [59.70, 66.32], default=60.40)
    return a_threshold.astype("float32"), c_threshold.astype("float32")


def make_boundary_targets(target_frame, base_frame):
    """Build official-rule endpoints directly from observed training traits."""
    traits = target_frame[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    windex = pd.to_numeric(target_frame["WINDEX"], errors="coerce").to_numpy(dtype="float32")
    a_threshold, c_threshold = _yield_thresholds(base_frame)
    ranks = _quality_ranks(traits)
    worst_rank = np.maximum.reduce(ranks).astype("float32")
    growth_high = (traits[:, 6] >= 8.0).astype("float32")
    return pd.DataFrame(
        np.column_stack([
            windex,
            windex - a_threshold,
            windex - c_threshold,
            *ranks,
            worst_rank,
            growth_high,
        ]),
        columns=BOUNDARY_TARGET_NAMES,
        index=target_frame.index,
        dtype="float32",
    )


def derive_boundary_proxy_features(predictions, index):
    predictions = np.asarray(predictions, dtype="float32").copy()
    predictions[:, 0] = np.clip(predictions[:, 0], 35.0, 85.0)
    predictions[:, 1:3] = np.clip(predictions[:, 1:3], -30.0, 30.0)
    predictions[:, 3:8] = np.clip(predictions[:, 3:8], 0.0, 4.0)
    predictions[:, 8] = np.clip(predictions[:, 8], 0.0, 1.0)
    return pd.DataFrame(
        predictions,
        columns=BOUNDARY_PROXY_FEATURE_NAMES,
        index=index,
        dtype="float32",
    )


def build_nested_boundary_features(
    x_train,
    x_validation,
    x_test,
    target_train,
    target_validation,
    farm_groups,
    categorical_features,
    task_type="CPU",
    inner_splits=2,
    random_seed=142,
    iterations=700,
):
    """Directly reconstruct official decision endpoints with inner farm OOF."""
    train_targets = make_boundary_targets(target_train, x_train)
    validation_targets = make_boundary_targets(target_validation, x_validation)
    train_required = target_train[TRAIT_COLUMNS + ["WINDEX"]].apply(pd.to_numeric, errors="coerce")
    validation_required = target_validation[TRAIT_COLUMNS + ["WINDEX"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_train = np.isfinite(train_required).all(axis=1) & (train_required > -90).all(axis=1)
    valid_validation = (
        np.isfinite(validation_required).all(axis=1) & (validation_required > -90).all(axis=1)
    )
    if valid_train.sum() < 10_000:
        raise ValueError(f"Too few valid rows for boundary proxy training: {valid_train.sum():,}")

    train_predictions = np.zeros((len(x_train), len(BOUNDARY_TARGET_NAMES)), dtype="float32")
    validation_predictions = np.zeros((len(x_validation), len(BOUNDARY_TARGET_NAMES)), dtype="float32")
    test_predictions = np.zeros((len(x_test), len(BOUNDARY_TARGET_NAMES)), dtype="float32")
    validation_assignment = np.arange(len(x_validation)) % inner_splits
    test_assignment = np.arange(len(x_test)) % inner_splits
    missing_positions = np.flatnonzero(~valid_train.to_numpy())
    missing_assignment = np.arange(len(missing_positions)) % inner_splits
    split_rows = np.flatnonzero(valid_train.to_numpy())
    split_groups = np.asarray(farm_groups)[split_rows]
    splitter = GroupKFold(n_splits=inner_splits)

    for inner_fold, (fit_local, hold_local) in enumerate(
        splitter.split(split_rows, groups=split_groups), start=1
    ):
        fit_rows = split_rows[fit_local]
        hold_rows = split_rows[hold_local]
        fit_target = train_targets.iloc[fit_rows].to_numpy(dtype="float32")
        means = fit_target.mean(axis=0)
        scales = np.maximum(fit_target.std(axis=0), 1e-3)
        standardized_fit = (fit_target - means) / scales
        standardized_hold = (
            train_targets.iloc[hold_rows].to_numpy(dtype="float32") - means
        ) / scales
        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=10.0,
            random_strength=0.5,
            random_seed=random_seed + inner_fold,
            task_type=task_type,
            allow_writing_files=False,
            verbose=100,
        )
        model.fit(
            x_train.iloc[fit_rows],
            standardized_fit,
            cat_features=categorical_features,
            eval_set=(x_train.iloc[hold_rows], standardized_hold),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        train_predictions[hold_rows] = model.predict(x_train.iloc[hold_rows]) * scales + means
        validation_rows = np.flatnonzero(validation_assignment == inner_fold - 1)
        test_rows = np.flatnonzero(test_assignment == inner_fold - 1)
        validation_predictions[validation_rows] = model.predict(
            x_validation.iloc[validation_rows]
        ) * scales + means
        test_predictions[test_rows] = model.predict(x_test.iloc[test_rows]) * scales + means
        assigned_missing = missing_positions[missing_assignment == inner_fold - 1]
        if len(assigned_missing):
            train_predictions[assigned_missing] = model.predict(
                x_train.iloc[assigned_missing]
            ) * scales + means
        del model
        gc.collect()

    metrics = []
    actual = validation_targets.loc[valid_validation].to_numpy(dtype="float32")
    predicted = validation_predictions[valid_validation.to_numpy()]
    for index, target in enumerate(BOUNDARY_TARGET_NAMES):
        metrics.append({
            "trait": target,
            "mae": mean_absolute_error(actual[:, index], predicted[:, index]),
            "r2": r2_score(actual[:, index], predicted[:, index]),
            "support": int(valid_validation.sum()),
            "proxy_type": "boundary",
        })
    return (
        derive_boundary_proxy_features(train_predictions, x_train.index),
        derive_boundary_proxy_features(validation_predictions, x_validation.index),
        derive_boundary_proxy_features(test_predictions, x_test.index),
        metrics,
    )


def build_nested_proxy_features(
    x_train,
    x_validation,
    x_test,
    target_train,
    target_validation,
    farm_groups,
    categorical_features,
    task_type="CPU",
    inner_splits=2,
    random_seed=42,
    iterations=900,
):
    """Return inner-OOF proxies with one-model predictions for every split."""
    target_train = target_train[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    target_validation = target_validation[TRAIT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_train = np.isfinite(target_train).all(axis=1) & (target_train > -90).all(axis=1)
    valid_validation = np.isfinite(target_validation).all(axis=1) & (target_validation > -90).all(axis=1)
    if valid_train.sum() < 10_000:
        raise ValueError(f"Too few valid rows for carcass proxy training: {valid_train.sum():,}")

    train_predictions = np.zeros((len(x_train), len(TRAIT_COLUMNS)), dtype="float32")
    validation_predictions = np.zeros((len(x_validation), len(TRAIT_COLUMNS)), dtype="float32")
    test_predictions = np.zeros((len(x_test), len(TRAIT_COLUMNS)), dtype="float32")
    validation_assignment = np.arange(len(x_validation)) % inner_splits
    test_assignment = np.arange(len(x_test)) % inner_splits
    missing_proxy_rows = ~valid_train.to_numpy()
    missing_positions = np.flatnonzero(missing_proxy_rows)
    missing_assignment = np.arange(len(missing_positions)) % inner_splits
    splitter = GroupKFold(n_splits=inner_splits)
    split_rows = np.flatnonzero(valid_train.to_numpy())
    split_groups = np.asarray(farm_groups)[split_rows]

    for inner_fold, (fit_local, hold_local) in enumerate(
        splitter.split(split_rows, groups=split_groups), start=1
    ):
        fit_rows = split_rows[fit_local]
        hold_rows = split_rows[hold_local]
        fit_target = target_train.iloc[fit_rows].to_numpy(dtype="float32")
        means = fit_target.mean(axis=0)
        scales = np.maximum(fit_target.std(axis=0), 1e-3)
        standardized_fit = (fit_target - means) / scales
        standardized_hold = (
            target_train.iloc[hold_rows].to_numpy(dtype="float32") - means
        ) / scales

        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            random_seed=random_seed + inner_fold,
            task_type=task_type,
            allow_writing_files=False,
            verbose=100,
        )
        model.fit(
            x_train.iloc[fit_rows],
            standardized_fit,
            cat_features=categorical_features,
            eval_set=(x_train.iloc[hold_rows], standardized_hold),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        train_predictions[hold_rows] = model.predict(x_train.iloc[hold_rows]) * scales + means
        validation_rows = np.flatnonzero(validation_assignment == inner_fold - 1)
        test_rows = np.flatnonzero(test_assignment == inner_fold - 1)
        validation_predictions[validation_rows] = (
            model.predict(x_validation.iloc[validation_rows]) * scales + means
        )
        test_predictions[test_rows] = model.predict(x_test.iloc[test_rows]) * scales + means
        assigned_missing = missing_positions[missing_assignment == inner_fold - 1]
        if len(assigned_missing):
            train_predictions[assigned_missing] = (
                model.predict(x_train.iloc[assigned_missing]) * scales + means
            )
        del model
        gc.collect()

    metrics = []
    validation_actual = target_validation.loc[valid_validation].to_numpy(dtype="float32")
    validation_predicted = validation_predictions[valid_validation.to_numpy()]
    for index, trait in enumerate(TRAIT_COLUMNS):
        metrics.append({
            "trait": trait,
            "mae": mean_absolute_error(validation_actual[:, index], validation_predicted[:, index]),
            "r2": r2_score(validation_actual[:, index], validation_predicted[:, index]),
            "support": int(valid_validation.sum()),
        })

    return (
        derive_proxy_features(train_predictions, x_train),
        derive_proxy_features(validation_predictions, x_validation),
        derive_proxy_features(test_predictions, x_test),
        metrics,
    )


# -----------------------------------------------------------------------------
# V13 training pipeline
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
VERSION = "v13"
BASE_DIR = Path(os.getenv("HANWOO_BASE_DIR", SCRIPT_DIR)).expanduser().resolve()
DATA_DIR = str(Path(os.getenv("HANWOO_DATA_DIR", BASE_DIR / "data")).expanduser().resolve())
OUT_DIR = str(Path(os.getenv("HANWOO_OUTPUT_DIR", BASE_DIR / "outputs" / VERSION)).expanduser().resolve())

N_SPLITS = 5
GROUP_SPECS = [("KPN_NO", "kpn"), ("stn", "stn"), ("sido", "sido"), ("sigungu", "sigungu")]
CORE_GROUP_SPECS = list(GROUP_SPECS)
EXTENDED_LINEAGE_SPECS = [
    ("FATHER_CATTLE_NO", "father"),
    ("MOTHER_ANIMAL_NO", "mother"),
    ("F_GMOTHER_ANIMAL_NO", "f_gmother"),
    ("F_GFATHER_CATTLE_NO", "f_gfather"),
    ("M_GMOTHER_ANIMAL_NO", "m_gmother"),
    ("M_GFATHER_CATTLE_NO", "m_gfather"),
]
ENABLE_EXTENDED_LINEAGE = True
if ENABLE_EXTENDED_LINEAGE:
    GROUP_SPECS += EXTENDED_LINEAGE_SPECS
ENABLE_B_SPECIALIST = False
ENABLE_YIELD_BALANCE = False
ENABLE_DUAL_C = True
ENABLE_PROXY_TRAITS = True
ENABLE_BOUNDARY_PROXY = False
ENABLE_BC_SPECIALIST = False
ENABLE_DRIFT_RESISTANT = False
ENABLE_REGIONAL_CALIBRATION = False
C_BALANCE_POWER = 0.25
CATBOOST_TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU")
XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cpu")
SAVE_OOF = os.getenv("SAVE_OOF", "1") == "1"
SMOKE_STAGE = os.getenv("PIPELINE_SMOKE_STAGE", os.getenv("V13_SMOKE_STAGE", "")).lower()


def resolve_data_path(env_name, candidates):
    """Find competition data files from env override or common local names."""
    override = os.getenv(env_name)
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"{env_name} points to a missing file: {path}")

    checked = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = Path(DATA_DIR) / candidate
        checked.append(str(path))
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find required data file. Checked: " + ", ".join(checked)
    )


def run_setup_smoke_check():
    """Validate required local data paths without loading competition data."""
    required = {
        "train": ("HANWOO_TRAIN_PATH", ["hanwoo_train.csv", "train_hanwoo.csv", "train.csv"]),
        "test": ("HANWOO_TEST_PATH", ["test_hanwoo.csv", "hanwoo_test.csv", "test.csv"]),
        "lineage": ("HANWOO_LINEAGE_PATH", ["hanwoo_lineage_0612.csv", "hanwoo_lineage.csv"]),
    }
    resolved = {}
    for label, (env_name, candidates) in required.items():
        resolved[label] = resolve_data_path(env_name, candidates)

    print(f"[{VERSION}] Setup smoke stage complete; required files resolve:")
    for label, path in resolved.items():
        print(f"[{VERSION}]   {label}: {path}")


def elapsed(start):
    return f"{(time.time() - start) / 60:.1f}min"


def normalized_keys(series):
    return series.astype("string").fillna("__MISSING__")


def attach_fold_features(train_fold, val_fold, test_fold=None):
    """Create target-free fold statistics and cohort-normalized physical features."""
    outputs = [train_fold.copy(), val_fold.copy()]
    if test_fold is not None:
        outputs.append(test_fold.copy())

    for group_col, prefix in GROUP_SPECS:
        if group_col not in train_fold.columns:
            continue
        train_keys = normalized_keys(train_fold[group_col])
        counts = train_keys.value_counts(dropna=False)
        for output_index, output in enumerate(outputs):
            values = normalized_keys(output[group_col]).map(counts).fillna(0).to_numpy(dtype="float32")
            if output_index == 0:
                values = np.maximum(values - 1.0, 0.0)
            output[f"{prefix}_group_count"] = values
            output[f"{prefix}_group_log_count"] = np.log1p(values).astype("float32")

    # Weight has different meaning across sex and slaughter age. These target-free
    # fold statistics express whether an animal is unusually heavy for its cohort.
    age_bin = (pd.to_numeric(train_fold["AGE"], errors="coerce").fillna(-1) // 2 * 2).astype(int)
    sex = normalized_keys(train_fold["sex_code"])
    cohort_source = pd.DataFrame(
        {"sex": sex, "age_bin": age_bin, "weight": pd.to_numeric(train_fold["WEIGHT"], errors="coerce")},
        index=train_fold.index,
    )
    cohort_stats = cohort_source.groupby(["sex", "age_bin"], dropna=False)["weight"].agg(
        cohort_weight_mean="mean",
        cohort_weight_std="std",
        cohort_weight_median="median",
        cohort_weight_q25=lambda values: values.quantile(0.25),
        cohort_weight_q75=lambda values: values.quantile(0.75),
        cohort_count="count",
    )
    global_weight = float(cohort_source["weight"].mean())
    global_std = float(cohort_source["weight"].std())

    for output in outputs:
        output_age_bin = (pd.to_numeric(output["AGE"], errors="coerce").fillna(-1) // 2 * 2).astype(int)
        output_sex = normalized_keys(output["sex_code"])
        lookup = pd.MultiIndex.from_arrays([output_sex, output_age_bin], names=["sex", "age_bin"])
        mapped = cohort_stats.reindex(lookup)
        weight = pd.to_numeric(output["WEIGHT"], errors="coerce").to_numpy(dtype="float32")
        mean = mapped["cohort_weight_mean"].fillna(global_weight).to_numpy(dtype="float32")
        std = mapped["cohort_weight_std"].fillna(global_std).clip(lower=5.0).to_numpy(dtype="float32")
        median = mapped["cohort_weight_median"].fillna(global_weight).to_numpy(dtype="float32")
        q25 = mapped["cohort_weight_q25"].fillna(global_weight - global_std).to_numpy(dtype="float32")
        q75 = mapped["cohort_weight_q75"].fillna(global_weight + global_std).to_numpy(dtype="float32")
        output["cohort_count"] = mapped["cohort_count"].fillna(0).to_numpy(dtype="float32")
        output["weight_cohort_residual"] = weight - mean
        output["weight_cohort_z"] = (weight - mean) / std
        output["weight_cohort_median_residual"] = weight - median
        output["weight_cohort_iqr_position"] = (weight - q25) / np.maximum(q75 - q25, 5.0)
        output["xgb_sex_code"] = pd.to_numeric(output["sex_code"].astype("string"), errors="coerce").fillna(-1).astype("float32")
        output["age_rearing_gap"] = (
            pd.to_numeric(output["AGE"], errors="coerce")
            - pd.to_numeric(output["rearing_months"], errors="coerce")
        ).astype("float32")

        rearing = pd.to_numeric(output["rearing_months"], errors="coerce").fillna(0).clip(lower=1)
        for source, target in [
            ("s_heat_3m", "heat_3m_per_rearing_month"),
            ("s_heat_6m", "heat_6m_per_rearing_month"),
            ("s_heat_12m", "heat_12m_per_rearing_month"),
        ]:
            if source in output.columns:
                output[target] = (pd.to_numeric(output[source], errors="coerce") / rearing).astype("float32")

        output["has_kpn"] = (normalized_keys(output["KPN_NO"]) != "__MISSING__").astype("float32")
        bv_columns = [column for column in output.columns if column.startswith("kpn_") and column.endswith("_bv")]
        if bv_columns:
            output["has_kpn_bv"] = output[bv_columns].notna().any(axis=1).astype("float32")
            output["kpn_bv_missing_count"] = output[bv_columns].isna().sum(axis=1).astype("float32")
        else:
            output["has_kpn_bv"] = np.float32(0.0)
            output["kpn_bv_missing_count"] = np.float32(0.0)
        weather_columns = [column for column in output.columns if column.startswith("s_")]
        output["weather_missing_count"] = output[weather_columns].isna().sum(axis=1).astype("float32")

    return outputs[0], outputs[1], outputs[2] if test_fold is not None else None


def prepare_features(frame, feature_cols, categorical_cols, category_levels):
    result = frame[feature_cols].copy()
    for column in categorical_cols:
        values = result[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(category_levels[column]), "__UNKNOWN__")
        result[column] = values.astype(pd.CategoricalDtype(categories=category_levels[column]))
    for column in feature_cols:
        if column not in categorical_cols:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(-999).astype("float32")
    return result


def prepare_catboost_features(frame, categorical_cols):
    result = frame.copy()
    for column in categorical_cols:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def prepare_xgboost_features(frame, feature_cols):
    return frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-999).astype("float32")


def lgb_macro_f1(y_true, probabilities):
    return "macro_f1", f1_score(y_true, np.argmax(probabilities, axis=1), average="macro"), True


def build_xgb_classifier(positive_rate, balance_power=0.5):
    # Mild square-root balancing makes the lower-frequency boundary visible
    # without destroying probability ranking as full inverse weighting can.
    scale_pos_weight = float(
        ((1.0 - positive_rate) / max(positive_rate, 1e-6)) ** balance_power
    )
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=1800,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        max_bin=256,
        tree_method="hist",
        device=XGBOOST_DEVICE,
        eval_metric="logloss",
        early_stopping_rounds=120,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


def combine_yield_hurdle(c_probability, a_given_non_c):
    """Reconstruct P(A), P(B), P(C) from ordered binary boundaries."""
    p_c = np.clip(c_probability, 1e-6, 1.0 - 1e-6)
    p_non_c = 1.0 - p_c
    p_a = p_non_c * np.clip(a_given_non_c, 1e-6, 1.0 - 1e-6)
    p_b = p_non_c - p_a
    result = np.column_stack([p_a, p_b, p_c]).astype("float32")
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def adjust_yield_probabilities(probabilities, b_probability=None, b_weight=0.0, c_multiplier=1.0):
    """Blend a B specialist and adjust C odds while preserving a valid simplex."""
    result = np.asarray(probabilities, dtype="float32").copy()
    if b_probability is not None and b_weight > 0:
        specialist_b = np.clip(np.asarray(b_probability, dtype="float32"), 1e-6, 1.0 - 1e-6)
        target_b = (1.0 - b_weight) * result[:, 1] + b_weight * specialist_b
        ac_total = np.maximum(result[:, 0] + result[:, 2], 1e-12)
        a_share = result[:, 0] / ac_total
        remaining = 1.0 - target_b
        result[:, 0] = remaining * a_share
        result[:, 1] = target_b
        result[:, 2] = remaining * (1.0 - a_share)
    result[:, 2] *= c_multiplier
    result = np.clip(result, 1e-8, None)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def blend_bc_specialist(probabilities, c_given_bc, specialist_weight):
    """Preserve P(A) and use a B/C-only model to redistribute P(B)+P(C)."""
    result = np.asarray(probabilities, dtype="float32").copy()
    if specialist_weight <= 0:
        return result
    bc_total = np.maximum(result[:, 1] + result[:, 2], 1e-12)
    base_c_share = result[:, 2] / bc_total
    specialist_c_share = np.clip(np.asarray(c_given_bc, dtype="float32"), 1e-6, 1.0 - 1e-6)
    c_share = (1.0 - specialist_weight) * base_c_share + specialist_weight * specialist_c_share
    result[:, 1] = bc_total * (1.0 - c_share)
    result[:, 2] = bc_total * c_share
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def is_drift_feature(column):
    """Features that made train/test source classification nearly deterministic."""
    exact = {
        "stn", "sido", "sigungu", "density", "death_cnt", "farm_size", "profile_year",
        "stn_group_count", "sido_group_count", "sigungu_group_count",
    }
    return column in exact or column.startswith(("s_", "weather_", "heat_"))


def cumulative_to_ordinal_probabilities(cumulative):
    """Convert P(rank <= k), k=0..3, into five ordered class probabilities."""
    cumulative = np.clip(np.asarray(cumulative, dtype="float32"), 1e-6, 1.0 - 1e-6)
    cumulative = np.maximum.accumulate(cumulative, axis=1)
    result = np.column_stack(
        [
            cumulative[:, 0],
            cumulative[:, 1] - cumulative[:, 0],
            cumulative[:, 2] - cumulative[:, 1],
            cumulative[:, 3] - cumulative[:, 2],
            1.0 - cumulative[:, 3],
        ]
    ).astype("float32")
    result = np.clip(result, 1e-8, 1.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def blend_quality_probabilities(cat_quality, ordinal_graded, ordinal_weight, q_encoder, graded_order):
    """Keep CatBoost's outlier gate and blend conditional graded quality probabilities."""
    outlier = "등외"
    q_index = {label: index for index, label in enumerate(q_encoder.classes_)}
    outlier_index = q_index[outlier]
    graded_indices = [q_index[label] for label in graded_order]
    outlier_probability = np.clip(cat_quality[:, outlier_index], 1e-6, 1.0 - 1e-6)
    cat_graded = cat_quality[:, graded_indices]
    cat_graded /= np.maximum(cat_graded.sum(axis=1, keepdims=True), 1e-12)
    conditional = (1.0 - ordinal_weight) * cat_graded + ordinal_weight * ordinal_graded
    result = np.zeros_like(cat_quality, dtype="float32")
    result[:, outlier_index] = outlier_probability
    result[:, graded_indices] = conditional * (1.0 - outlier_probability).reshape(-1, 1)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def combine_hierarchical_probabilities(q_proba, y_proba, q_encoder, y_encoder, final_encoder):
    result = np.zeros((len(q_proba), len(final_encoder.classes_)), dtype="float32")
    qi = {label: i for i, label in enumerate(q_encoder.classes_)}
    yi = {label: i for i, label in enumerate(y_encoder.classes_)}
    fi = {label: i for i, label in enumerate(final_encoder.classes_)}
    outlier = "등외"
    for grade in final_encoder.classes_:
        if grade == outlier:
            result[:, fi[grade]] = q_proba[:, qi[outlier]]
        else:
            result[:, fi[grade]] = q_proba[:, qi[grade[:-1]]] * y_proba[:, yi[grade[-1]]]
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def apply_class_scales(probabilities, scales):
    result = probabilities * scales.reshape(1, -1)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def optimize_class_scales(y_true, probabilities, max_rows=250_000):
    if len(y_true) > max_rows:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(y_true), max_rows, replace=False)
        y_search, p_search = y_true[indices], probabilities[indices]
    else:
        y_search, p_search = y_true, probabilities
    scales = np.ones(probabilities.shape[1], dtype="float32")
    factors = [0.85, 0.925, 1.0, 1.08, 1.18]
    best = f1_score(y_search, np.argmax(p_search, axis=1), average="macro")
    for _ in range(2):
        improved = False
        for cls in range(len(scales)):
            current = scales[cls]
            class_best = current
            for candidate in np.unique(np.clip(current * np.asarray(factors), 0.4, 2.5)):
                trial = scales.copy()
                trial[cls] = candidate
                score = f1_score(y_search, np.argmax(p_search * trial, axis=1), average="macro")
                if score > best + 1e-7:
                    best, class_best = score, candidate
            if class_best != current:
                scales[cls] = class_best
                improved = True
        if not improved:
            break
    return scales


def blend_probabilities(direct, hierarchical, direct_weight, mode):
    if mode == "geometric":
        result = np.exp(
            direct_weight * np.log(np.clip(direct, 1e-8, 1.0))
            + (1.0 - direct_weight) * np.log(np.clip(hierarchical, 1e-8, 1.0))
        )
        return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)
    return direct_weight * direct + (1.0 - direct_weight) * hierarchical


def fit_calibrator(y_true, direct, hierarchical, resilient=None):
    direct_scales = optimize_class_scales(y_true, direct)
    hierarchical_scales = optimize_class_scales(y_true, hierarchical)
    direct_cal = apply_class_scales(direct, direct_scales)
    hierarchical_cal = apply_class_scales(hierarchical, hierarchical_scales)
    resilient_scales = np.ones(direct.shape[1], dtype="float32")
    resilient_weight = 0.0
    resilient_mode = "arithmetic"
    direct_candidates = [(0.0, "arithmetic")]
    if resilient is not None:
        resilient_scales = optimize_class_scales(y_true, resilient)
        resilient_cal = apply_class_scales(resilient, resilient_scales)
        direct_candidates = []
        for resilient_mode_candidate in ["arithmetic", "geometric"]:
            for resilient_weight_candidate in [0.0, 0.1, 0.2, 0.3, 0.5]:
                direct_candidates.append((resilient_weight_candidate, resilient_mode_candidate))
    if resilient is not None and len(y_true) > 250_000:
        rng = np.random.default_rng(314)
        score_indices = rng.choice(len(y_true), 250_000, replace=False)
    else:
        score_indices = np.arange(len(y_true))
    candidates = []
    for resilient_weight_candidate, resilient_mode_candidate in direct_candidates:
        if resilient is None or resilient_weight_candidate <= 0:
            combined_direct = direct_cal[score_indices]
        else:
            combined_direct = blend_probabilities(
                resilient_cal[score_indices],
                direct_cal[score_indices],
                resilient_weight_candidate,
                resilient_mode_candidate,
            )
        for mode in ["arithmetic", "geometric"]:
            for weight in np.linspace(0.0, 1.0, 21):
                probabilities = blend_probabilities(
                    combined_direct, hierarchical_cal[score_indices], weight, mode
                )
                candidates.append((
                    f1_score(y_true[score_indices], probabilities.argmax(1), average="macro"),
                    mode,
                    weight,
                    resilient_weight_candidate,
                    resilient_mode_candidate,
                ))
    score, mode, weight, resilient_weight, resilient_mode = max(
        candidates, key=lambda row: row[0]
    )
    return {
        "direct_scales": direct_scales,
        "hierarchical_scales": hierarchical_scales,
        "resilient_scales": resilient_scales,
        "resilient_weight": resilient_weight,
        "resilient_mode": resilient_mode,
        "mode": mode,
        "direct_weight": weight,
        "fit_score": score,
    }


def apply_calibrator(direct, hierarchical, calibrator, resilient=None):
    calibrated_direct = apply_class_scales(direct, calibrator["direct_scales"])
    if resilient is not None and calibrator.get("resilient_weight", 0.0) > 0:
        calibrated_direct = blend_probabilities(
            apply_class_scales(resilient, calibrator["resilient_scales"]),
            calibrated_direct,
            calibrator["resilient_weight"],
            calibrator["resilient_mode"],
        )
    return blend_probabilities(
        calibrated_direct,
        apply_class_scales(hierarchical, calibrator["hierarchical_scales"]),
        calibrator["direct_weight"],
        calibrator["mode"],
    )


def fit_regional_priors(y_true, regions, n_classes, alpha=2000.0):
    """Smoothed class-prior ratios for region-aware probability correction."""
    region_values = normalized_keys(pd.Series(regions)).reset_index(drop=True)
    global_counts = np.bincount(y_true, minlength=n_classes).astype("float64")
    global_prior = global_counts / np.maximum(global_counts.sum(), 1.0)
    frame = pd.DataFrame({"region": region_values, "target": y_true})
    counts = pd.crosstab(frame["region"], frame["target"]).reindex(
        columns=np.arange(n_classes), fill_value=0
    )
    posterior = (
        counts.to_numpy(dtype="float64") + alpha * global_prior.reshape(1, -1)
    ) / (counts.sum(axis=1).to_numpy(dtype="float64").reshape(-1, 1) + alpha)
    ratios = posterior / np.maximum(global_prior.reshape(1, -1), 1e-12)
    return pd.DataFrame(ratios.astype("float32"), index=counts.index), global_prior


def apply_regional_priors(probabilities, regions, ratio_table, beta):
    if beta <= 0:
        return np.asarray(probabilities, dtype="float32")
    region_values = normalized_keys(pd.Series(regions)).reset_index(drop=True)
    ratios = ratio_table.reindex(region_values).fillna(1.0).to_numpy(dtype="float32")
    result = np.asarray(probabilities, dtype="float32") * np.power(ratios, beta)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def save_class_metrics(y_true, probabilities, labels, path):
    predicted = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=np.arange(len(labels)), zero_division=0
    )
    pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}).to_csv(path, index=False)


def main():
    print(f"\n[{VERSION}] Starting specialized three-model pipeline...")
    print(
        f"[{VERSION}] CatBoost={CATBOOST_TASK_TYPE}, XGBoost={XGBOOST_DEVICE}, "
            f"extended_lineage={ENABLE_EXTENDED_LINEAGE}, b_specialist={ENABLE_B_SPECIALIST}, "
            f"yield_balance={ENABLE_YIELD_BALANCE}, dual_c={ENABLE_DUAL_C}, "
            f"proxy_traits={ENABLE_PROXY_TRAITS}, boundary_proxy={ENABLE_BOUNDARY_PROXY}, "
            f"bc_specialist={ENABLE_BC_SPECIALIST}, drift_resistant={ENABLE_DRIFT_RESISTANT}, "
            f"regional_calibration={ENABLE_REGIONAL_CALIBRATION}, c_power={C_BALANCE_POWER}, "
        f"smoke={SMOKE_STAGE or 'off'}"
    )
    if SMOKE_STAGE in {"check", "imports", "config"}:
        print(f"[{VERSION}] Import/config check complete; no data files were read.")
        raise SystemExit(0)
    if SMOKE_STAGE == "setup":
        try:
            run_setup_smoke_check()
        except FileNotFoundError as exc:
            print(f"[{VERSION}] Missing required data: {exc}")
            raise SystemExit(2) from None
        else:
            raise SystemExit(0)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    started = time.time()
    processor_class = HanwooDataProcessorV12 if ENABLE_EXTENDED_LINEAGE else HanwooDataProcessorV8
    processor = processor_class(DATA_DIR)
    processor.load_auxiliary_data()
    train_path = resolve_data_path("HANWOO_TRAIN_PATH", ["hanwoo_train.csv", "train_hanwoo.csv", "train.csv"])
    test_path = resolve_data_path("HANWOO_TEST_PATH", ["test_hanwoo.csv", "hanwoo_test.csv", "test.csv"])
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    train = processor.transform(train_raw, is_train=True)
    test_base = processor.transform(test_raw, is_train=False)
    del train_raw
    gc.collect()

    final_encoder = LabelEncoder().fit(processor.FINAL_GRADES)
    q_encoder = LabelEncoder().fit(processor.QUALITY_ORDER)
    y_encoder = LabelEncoder().fit(processor.YIELD_ORDER)
    y_final = final_encoder.transform(train["LAST_GRADE"].astype(str))
    y_quality = q_encoder.transform(train["target_q"].astype(str))
    y_yield = y_encoder.transform(train["target_y"].astype(str))
    is_graded = train["target_q"].astype(str).to_numpy() != "등외"
    graded_quality_order = ["1++", "1+", "1", "2", "3"]
    quality_rank_map = {label: rank for rank, label in enumerate(graded_quality_order)}
    y_quality_rank = train["target_q"].astype(str).map(quality_rank_map).fillna(-1).to_numpy(dtype="int8")
    groups = train["FARM_UNIQUE_NO"].astype(str)

    # A small prototype is enough to discover generated column names and dtypes.
    # Building fold aggregates on all 2.4M rows here would duplicate work before CV.
    prototype_source = train.iloc[: min(10_000, len(train))].copy()
    prototype, _, _ = attach_fold_features(prototype_source, prototype_source.iloc[:0].copy())
    target_related = {"BACKFAT", "REA", "WINDEX", "WGRADE", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH", "COST_AMT"}
    non_features = target_related | {"LAST_GRADE", "target_q", "target_y", "FARM_UNIQUE_NO", "grade_score"}
    generated_features = [column for column in prototype.columns if column not in train.columns]
    base_features = [
        column for column in train.columns
        if column not in non_features
        and column in test_base.columns
        and (is_numeric_dtype(train[column]) or isinstance(train[column].dtype, pd.CategoricalDtype))
    ]
    extended_lineage_columns = {column for column, _ in EXTENDED_LINEAGE_SPECS}
    lgb_features = [column for column in base_features if column not in extended_lineage_columns]
    lgb_features += [f"{prefix}_group_count" for _, prefix in CORE_GROUP_SPECS]

    cat_quality_features = list(dict.fromkeys(
        base_features
        + [f"{prefix}_group_count" for _, prefix in GROUP_SPECS]
        + [column for column in generated_features if column.startswith(("has_", "weather_missing"))]
    ))
    categorical_features = [column for column in processor.CATEGORICAL_COLUMNS if column in cat_quality_features]

    xgb_numeric_prefixes = (
        "s_", "kpn_", "weight_", "cohort_", "heat_", "age_", "has_", "weather_",
    )
    xgb_explicit = {
        "WEIGHT", "AGE", "rearing_months", "density", "death_cnt", "abatt_year", "birth_year",
        "abatt_month_sin", "abatt_month_cos", "birth_month_sin", "birth_month_cos", "xgb_sex_code",
    }
    xgb_yield_features = [
        column for column in prototype.columns
        if column not in non_features
        and is_numeric_dtype(prototype[column])
        and (column in xgb_explicit or column.startswith(xgb_numeric_prefixes) or column.endswith("_group_count"))
    ]
    if ENABLE_PROXY_TRAITS:
        cat_quality_features += list(PROXY_FEATURE_NAMES)
        xgb_yield_features += list(PROXY_FEATURE_NAMES)
    if ENABLE_BOUNDARY_PROXY:
        cat_quality_features += list(BOUNDARY_PROXY_FEATURE_NAMES)
        xgb_yield_features += list(BOUNDARY_PROXY_FEATURE_NAMES)

    lgb_features = list(dict.fromkeys(lgb_features))
    cat_quality_features = list(dict.fromkeys(cat_quality_features))
    xgb_yield_features = list(dict.fromkeys(xgb_yield_features))
    drift_lgb_features = [column for column in lgb_features if not is_drift_feature(column)]

    category_levels = {
        column: sorted(set(train[column].astype("string").fillna("__MISSING__").unique()) | {"__MISSING__", "__UNKNOWN__"})
        for column in categorical_features
    }
    del prototype, prototype_source
    gc.collect()
    print(
        f"[{VERSION}] Feature views: LightGBM={len(lgb_features)}, "
        f"CatBoost-quality={len(cat_quality_features)}, XGBoost-yield={len(xgb_yield_features)}, "
        f"drift-LightGBM={len(drift_lgb_features) if ENABLE_DRIFT_RESISTANT else 0}"
    )

    lgb_params = dict(
        objective="multiclass", metric="None", learning_rate=0.03, num_leaves=127,
        min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=2.0, max_depth=12, random_state=42, verbose=-1,
        n_estimators=3000,
    )
    cat_params = dict(
        loss_function="MultiClass", eval_metric="MultiClass", iterations=1800,
        learning_rate=0.04, depth=8, l2_leaf_reg=8.0, random_strength=0.75,
        random_seed=42, task_type=CATBOOST_TASK_TYPE, allow_writing_files=False, verbose=100,
    )
    ordinal_lgb_params = dict(
        objective="binary", metric="binary_logloss", learning_rate=0.04, num_leaves=63,
        min_child_samples=150, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.2, lambda_l2=4.0, max_depth=10, random_state=42, verbose=-1,
        n_estimators=1800,
    )

    n_rows, n_test, n_classes = len(train), len(test_base), len(final_encoder.classes_)
    direct_oof = np.zeros((n_rows, n_classes), dtype="float32")
    hierarchical_oof = np.zeros_like(direct_oof)
    quality_oof = np.zeros((n_rows, len(q_encoder.classes_)), dtype="float32")
    ordinal_quality_oof = np.zeros((n_rows, len(graded_quality_order)), dtype="float32")
    yield_oof = np.zeros((n_rows, len(y_encoder.classes_)), dtype="float32")
    yield_alt_oof = np.zeros_like(yield_oof)
    b_specialist_oof = np.zeros(n_rows, dtype="float32")
    bc_specialist_oof = np.zeros(n_rows, dtype="float32")
    drift_direct_oof = np.zeros_like(direct_oof)
    direct_test = np.zeros((n_test, n_classes), dtype="float32")
    hierarchical_test = np.zeros_like(direct_test)
    quality_test_average = np.zeros((n_test, len(q_encoder.classes_)), dtype="float32")
    ordinal_quality_test = np.zeros((n_test, len(graded_quality_order)), dtype="float32")
    yield_test_average = np.zeros((n_test, len(y_encoder.classes_)), dtype="float32")
    yield_alt_test_average = np.zeros_like(yield_test_average)
    b_specialist_test_average = np.zeros(n_test, dtype="float32")
    bc_specialist_test_average = np.zeros(n_test, dtype="float32")
    drift_direct_test = np.zeros_like(direct_test)
    fold_ids = np.full(n_rows, -1, dtype="int8")
    importance = np.zeros(len(lgb_features), dtype="float64")
    component_rows = []
    proxy_metric_rows = []
    component_count = 4 + int(ENABLE_B_SPECIALIST) + int(ENABLE_BC_SPECIALIST) + int(ENABLE_DRIFT_RESISTANT)

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(train, y_final, groups=groups)):
        print(f"\n[Fold {fold + 1}/{N_SPLITS}] Building target-free feature views...")
        tr_aug, val_aug, test_aug = attach_fold_features(train.iloc[tr_idx], train.iloc[val_idx], test_base)
        all_features = list(dict.fromkeys(lgb_features + cat_quality_features + xgb_yield_features))
        proxy_output_names = set(PROXY_FEATURE_NAMES) | set(BOUNDARY_PROXY_FEATURE_NAMES)
        initial_features = [column for column in all_features if column not in proxy_output_names]
        x_tr = prepare_features(tr_aug, initial_features, categorical_features, category_levels)
        x_val = prepare_features(val_aug, initial_features, categorical_features, category_levels)
        x_test = prepare_features(test_aug, initial_features, categorical_features, category_levels)
        fold_ids[val_idx] = fold

        if ENABLE_PROXY_TRAITS or ENABLE_BOUNDARY_PROXY:
            proxy_input_features = [
                column for column in cat_quality_features if column not in proxy_output_names
            ]
            proxy_input_features = list(dict.fromkeys(proxy_input_features + ["xgb_sex_code"]))
            proxy_categorical = [
                column for column in categorical_features if column in proxy_input_features
            ]
            proxy_train_input = prepare_catboost_features(
                x_tr[proxy_input_features], proxy_categorical
            )
            proxy_val_input = prepare_catboost_features(
                x_val[proxy_input_features], proxy_categorical
            )
            proxy_test_input = prepare_catboost_features(
                x_test[proxy_input_features], proxy_categorical
            )
            if ENABLE_PROXY_TRAITS:
                print("  [proxy] Nested 2-Fold multi-target carcass reconstruction")
                proxy_train, proxy_val, proxy_test, proxy_metrics = build_nested_proxy_features(
                    proxy_train_input,
                    proxy_val_input,
                    proxy_test_input,
                    train.iloc[tr_idx][TRAIT_COLUMNS],
                    train.iloc[val_idx][TRAIT_COLUMNS],
                    groups.iloc[tr_idx].to_numpy(),
                    proxy_categorical,
                    task_type=CATBOOST_TASK_TYPE,
                    inner_splits=2,
                    random_seed=42 + fold * 10,
                )
                for column in PROXY_FEATURE_NAMES:
                    x_tr[column] = proxy_train[column].to_numpy(dtype="float32")
                    x_val[column] = proxy_val[column].to_numpy(dtype="float32")
                    x_test[column] = proxy_test[column].to_numpy(dtype="float32")
                for metric in proxy_metrics:
                    proxy_metric_rows.append({"fold": fold + 1, "proxy_type": "trait", **metric})
                    print(
                        f"    {metric['trait']}: MAE={metric['mae']:.4f}, "
                        f"R2={metric['r2']:.4f}"
                    )
                del proxy_train, proxy_val, proxy_test, proxy_metrics
            if ENABLE_BOUNDARY_PROXY:
                print("  [boundary] Nested 2-Fold direct rule-endpoint reconstruction")
                boundary_train, boundary_val, boundary_test, boundary_metrics = build_nested_boundary_features(
                    proxy_train_input,
                    proxy_val_input,
                    proxy_test_input,
                    train.iloc[tr_idx][TRAIT_COLUMNS + ["WINDEX"]],
                    train.iloc[val_idx][TRAIT_COLUMNS + ["WINDEX"]],
                    groups.iloc[tr_idx].to_numpy(),
                    proxy_categorical,
                    task_type=CATBOOST_TASK_TYPE,
                    inner_splits=2,
                    random_seed=142 + fold * 10,
                )
                for column in BOUNDARY_PROXY_FEATURE_NAMES:
                    x_tr[column] = boundary_train[column].to_numpy(dtype="float32")
                    x_val[column] = boundary_val[column].to_numpy(dtype="float32")
                    x_test[column] = boundary_test[column].to_numpy(dtype="float32")
                for metric in boundary_metrics:
                    proxy_metric_rows.append({"fold": fold + 1, **metric})
                    print(
                        f"    {metric['trait']}: MAE={metric['mae']:.4f}, "
                        f"R2={metric['r2']:.4f}"
                    )
                del boundary_train, boundary_val, boundary_test, boundary_metrics
            del proxy_train_input, proxy_val_input, proxy_test_input
            gc.collect()

        print(f"  [1/{component_count}] LightGBM direct 16-class model")
        direct_model = lgb.LGBMClassifier(**lgb_params, num_class=n_classes)
        direct_model.fit(
            x_tr[lgb_features], y_final[tr_idx],
            eval_set=[(x_val[lgb_features], y_final[val_idx])], eval_metric=lgb_macro_f1,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
        )
        direct_val = direct_model.predict_proba(x_val[lgb_features]).astype("float32")
        direct_oof[val_idx] = direct_val
        direct_test += direct_model.predict_proba(x_test[lgb_features]).astype("float32") / N_SPLITS
        importance += direct_model.feature_importances_ / N_SPLITS
        direct_f1 = f1_score(y_final[val_idx], direct_val.argmax(1), average="macro")
        print(f"  Direct Macro F1: {direct_f1:.4f}")
        del direct_model
        gc.collect()
        drift_direct_f1 = np.nan
        if ENABLE_DRIFT_RESISTANT:
            print("  [drift] LightGBM 16-class model without regional/farm/weather features")
            drift_params = {**lgb_params, "num_leaves": 95, "max_depth": 11, "lambda_l2": 4.0}
            drift_model = lgb.LGBMClassifier(**drift_params, num_class=n_classes)
            drift_model.fit(
                x_tr[drift_lgb_features], y_final[tr_idx],
                eval_set=[(x_val[drift_lgb_features], y_final[val_idx])], eval_metric=lgb_macro_f1,
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
            )
            drift_val = drift_model.predict_proba(x_val[drift_lgb_features]).astype("float32")
            drift_direct_oof[val_idx] = drift_val
            drift_direct_test += (
                drift_model.predict_proba(x_test[drift_lgb_features]).astype("float32") / N_SPLITS
            )
            drift_direct_f1 = f1_score(y_final[val_idx], drift_val.argmax(1), average="macro")
            print(f"  Drift-resistant Direct Macro F1: {drift_direct_f1:.4f}")
            del drift_model, drift_val
            gc.collect()
        if SMOKE_STAGE == "lgb":
            print(f"[{VERSION}] LightGBM smoke stage complete.")
            raise SystemExit(0)

        tr_graded = is_graded[tr_idx]
        val_graded = is_graded[val_idx]

        print(f"  [2/{component_count}] CatBoost quality/outlier model")
        x_tr_cat = prepare_catboost_features(x_tr[cat_quality_features], categorical_features)
        x_val_cat = prepare_catboost_features(x_val[cat_quality_features], categorical_features)
        x_test_cat = prepare_catboost_features(x_test[cat_quality_features], categorical_features)
        quality_model = CatBoostClassifier(**cat_params)
        quality_model.fit(
            x_tr_cat, y_quality[tr_idx], cat_features=categorical_features,
            eval_set=(x_val_cat, y_quality[val_idx]), early_stopping_rounds=150, use_best_model=True,
        )
        quality_val = quality_model.predict_proba(x_val_cat).astype("float32")
        quality_test = quality_model.predict_proba(x_test_cat).astype("float32")
        quality_oof[val_idx] = quality_val
        quality_test_average += quality_test / N_SPLITS
        quality_f1 = f1_score(y_quality[val_idx], quality_val.argmax(1), average="macro")
        print(f"  Quality Macro F1: {quality_f1:.4f}")
        del quality_model, x_tr_cat, x_val_cat, x_test_cat
        gc.collect()

        print(f"  [3/{component_count}] LightGBM ordinal quality boundaries")
        ordinal_val_cumulative = np.zeros((len(val_idx), 4), dtype="float32")
        ordinal_test_cumulative = np.zeros((n_test, 4), dtype="float32")
        for threshold in range(4):
            ordinal_train_target = (y_quality_rank[tr_idx][tr_graded] <= threshold).astype("int8")
            ordinal_val_target = (y_quality_rank[val_idx][val_graded] <= threshold).astype("int8")
            ordinal_model = lgb.LGBMClassifier(**ordinal_lgb_params)
            ordinal_model.fit(
                x_tr.loc[tr_graded, lgb_features],
                ordinal_train_target,
                eval_set=[(x_val.loc[val_graded, lgb_features], ordinal_val_target)],
                callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
            )
            ordinal_val_cumulative[:, threshold] = ordinal_model.predict_proba(x_val[lgb_features])[:, 1]
            ordinal_test_cumulative[:, threshold] = ordinal_model.predict_proba(x_test[lgb_features])[:, 1]
            del ordinal_model
            gc.collect()
        ordinal_val = cumulative_to_ordinal_probabilities(ordinal_val_cumulative)
        ordinal_test = cumulative_to_ordinal_probabilities(ordinal_test_cumulative)
        ordinal_quality_oof[val_idx] = ordinal_val
        ordinal_quality_test += ordinal_test / N_SPLITS
        ordinal_quality_f1 = f1_score(
            y_quality_rank[val_idx][val_graded], ordinal_val[val_graded].argmax(1), average="macro"
        )
        print(f"  Ordinal quality Macro F1: {ordinal_quality_f1:.4f}")

        print(f"  [4/{component_count}] XGBoost ordinal yield hurdle")
        x_tr_xgb = prepare_xgboost_features(x_tr, xgb_yield_features)
        x_val_xgb = prepare_xgboost_features(x_val, xgb_yield_features)
        x_test_xgb = prepare_xgboost_features(x_test, xgb_yield_features)
        c_index = int(np.flatnonzero(y_encoder.classes_ == "C")[0])
        a_index = int(np.flatnonzero(y_encoder.classes_ == "A")[0])
        b_index = int(np.flatnonzero(y_encoder.classes_ == "B")[0])

        c_train = (y_yield[tr_idx][tr_graded] == c_index).astype("int8")
        c_val = (y_yield[val_idx][val_graded] == c_index).astype("int8")
        c_model = build_xgb_classifier(float(c_train.mean()), balance_power=C_BALANCE_POWER)
        c_model.fit(
            x_tr_xgb.loc[tr_graded], c_train,
            eval_set=[(x_val_xgb.loc[val_graded], c_val)], verbose=100,
        )
        c_val_probability = c_model.predict_proba(x_val_xgb)[:, 1]
        c_test_probability = c_model.predict_proba(x_test_xgb)[:, 1]
        del c_model
        gc.collect()

        c_alt_val_probability = c_val_probability
        c_alt_test_probability = c_test_probability
        if ENABLE_DUAL_C:
            print("    Training aggressive-C companion (balance power=0.50)")
            c_alt_model = build_xgb_classifier(float(c_train.mean()), balance_power=0.5)
            c_alt_model.fit(
                x_tr_xgb.loc[tr_graded], c_train,
                eval_set=[(x_val_xgb.loc[val_graded], c_val)], verbose=100,
            )
            c_alt_val_probability = c_alt_model.predict_proba(x_val_xgb)[:, 1]
            c_alt_test_probability = c_alt_model.predict_proba(x_test_xgb)[:, 1]
            del c_alt_model
            gc.collect()

        tr_ab = tr_graded & (y_yield[tr_idx] != c_index)
        val_ab = val_graded & (y_yield[val_idx] != c_index)
        a_train = (y_yield[tr_idx][tr_ab] == a_index).astype("int8")
        a_val = (y_yield[val_idx][val_ab] == a_index).astype("int8")
        ab_model = build_xgb_classifier(float(a_train.mean()))
        ab_model.fit(
            x_tr_xgb.loc[tr_ab], a_train,
            eval_set=[(x_val_xgb.loc[val_ab], a_val)], verbose=100,
        )
        a_val_probability = ab_model.predict_proba(x_val_xgb)[:, 1]
        a_test_probability = ab_model.predict_proba(x_test_xgb)[:, 1]
        del ab_model
        gc.collect()

        yield_val = combine_yield_hurdle(c_val_probability, a_val_probability)
        yield_test = combine_yield_hurdle(c_test_probability, a_test_probability)
        yield_alt_val = combine_yield_hurdle(c_alt_val_probability, a_val_probability)
        yield_alt_test = combine_yield_hurdle(c_alt_test_probability, a_test_probability)
        yield_oof[val_idx] = yield_val
        yield_alt_oof[val_idx] = yield_alt_val
        yield_test_average += yield_test / N_SPLITS
        yield_alt_test_average += yield_alt_test / N_SPLITS
        yield_f1 = f1_score(y_yield[val_idx][val_graded], yield_val[val_graded].argmax(1), average="macro")
        c_recall = float(np.mean(yield_val[val_graded][c_val == 1].argmax(1) == c_index))
        print(f"  Yield Macro F1: {yield_f1:.4f}; C recall: {c_recall:.4f}")

        bc_specialist_f1 = np.nan
        bc_val_probability = np.zeros(len(val_idx), dtype="float32")
        if ENABLE_BC_SPECIALIST:
            print("  [B/C] XGBoost C-vs-B pairwise specialist")
            tr_bc = tr_graded & (y_yield[tr_idx] != a_index)
            val_bc = val_graded & (y_yield[val_idx] != a_index)
            bc_train = (y_yield[tr_idx][tr_bc] == c_index).astype("int8")
            bc_val = (y_yield[val_idx][val_bc] == c_index).astype("int8")
            bc_model = build_xgb_classifier(float(bc_train.mean()), balance_power=0.25)
            bc_model.fit(
                x_tr_xgb.loc[tr_bc], bc_train,
                eval_set=[(x_val_xgb.loc[val_bc], bc_val)], verbose=100,
            )
            bc_val_probability = bc_model.predict_proba(x_val_xgb)[:, 1].astype("float32")
            bc_test_probability = bc_model.predict_proba(x_test_xgb)[:, 1].astype("float32")
            bc_specialist_oof[val_idx] = bc_val_probability
            bc_specialist_test_average += bc_test_probability / N_SPLITS
            bc_specialist_f1 = f1_score(
                bc_val, (bc_val_probability[val_bc] >= 0.5).astype("int8"), average="binary"
            )
            print(f"  B/C specialist binary F1: {bc_specialist_f1:.4f}")
            del bc_model
            gc.collect()

        b_specialist_f1 = np.nan
        b_val_probability = np.zeros(len(val_idx), dtype="float32")
        if ENABLE_B_SPECIALIST:
            print(f"  [5/{component_count}] XGBoost B vs non-B specialist")
            b_train = (y_yield[tr_idx][tr_graded] == b_index).astype("int8")
            b_val = (y_yield[val_idx][val_graded] == b_index).astype("int8")
            b_model = build_xgb_classifier(float(b_train.mean()), balance_power=0.0)
            b_model.fit(
                x_tr_xgb.loc[tr_graded], b_train,
                eval_set=[(x_val_xgb.loc[val_graded], b_val)], verbose=100,
            )
            b_val_probability = b_model.predict_proba(x_val_xgb)[:, 1].astype("float32")
            b_test_probability = b_model.predict_proba(x_test_xgb)[:, 1].astype("float32")
            b_specialist_oof[val_idx] = b_val_probability
            b_specialist_test_average += b_test_probability / N_SPLITS
            b_specialist_f1 = f1_score(
                b_val, (b_val_probability[val_graded] >= 0.5).astype("int8"), average="binary"
            )
            print(f"  B specialist F1: {b_specialist_f1:.4f}")
            del b_model
            gc.collect()

        cat_hierarchical_val = combine_hierarchical_probabilities(
            quality_val, yield_val, q_encoder, y_encoder, final_encoder
        )
        ordinal_quality_val = blend_quality_probabilities(
            quality_val, ordinal_val, 1.0, q_encoder, graded_quality_order
        )
        ordinal_hierarchical_val = combine_hierarchical_probabilities(
            ordinal_quality_val, yield_val, q_encoder, y_encoder, final_encoder
        )
        cat_hierarchical_f1 = f1_score(
            y_final[val_idx], cat_hierarchical_val.argmax(1), average="macro"
        )
        ordinal_hierarchical_f1 = f1_score(
            y_final[val_idx], ordinal_hierarchical_val.argmax(1), average="macro"
        )

        diagnostic_hierarchical_f1 = -1.0
        diagnostic_ordinal_weight = 0.0
        diagnostic_b_weight = 0.0
        diagnostic_c_multiplier = 1.0
        diagnostic_alt_c_weight = 0.0
        diagnostic_bc_weight = 0.0
        diagnostic_ordinal_grid = np.linspace(0.0, 1.0, 6) if ENABLE_YIELD_BALANCE else np.linspace(0.0, 1.0, 11)
        diagnostic_b_grid = [0.0, 0.25, 0.5, 0.75] if ENABLE_B_SPECIALIST else [0.0]
        diagnostic_c_grid = [0.8, 0.9, 1.0] if ENABLE_YIELD_BALANCE else [1.0]
        diagnostic_alt_c_grid = np.linspace(0.0, 1.0, 6) if ENABLE_DUAL_C else [0.0]
        diagnostic_bc_grid = np.linspace(0.0, 1.0, 5) if ENABLE_BC_SPECIALIST else [0.0]
        for ordinal_weight in diagnostic_ordinal_grid:
            blended_quality = blend_quality_probabilities(
                quality_val, ordinal_val, ordinal_weight, q_encoder, graded_quality_order
            )
            for alt_c_weight in diagnostic_alt_c_grid:
                blended_yield = (
                    (1.0 - alt_c_weight) * yield_val + alt_c_weight * yield_alt_val
                )
                for b_weight in diagnostic_b_grid:
                    for c_multiplier in diagnostic_c_grid:
                        adjusted_yield = adjust_yield_probabilities(
                            blended_yield, b_val_probability, b_weight, c_multiplier
                        )
                        for bc_weight in diagnostic_bc_grid:
                            diagnostic_yield_candidate = blend_bc_specialist(
                                adjusted_yield, bc_val_probability, bc_weight
                            )
                            candidate_hierarchy = combine_hierarchical_probabilities(
                                blended_quality, diagnostic_yield_candidate,
                                q_encoder, y_encoder, final_encoder,
                            )
                            candidate_score = f1_score(
                                y_final[val_idx], candidate_hierarchy.argmax(1), average="macro"
                            )
                            if candidate_score > diagnostic_hierarchical_f1:
                                diagnostic_hierarchical_f1 = candidate_score
                                diagnostic_ordinal_weight = ordinal_weight
                                diagnostic_b_weight = b_weight
                                diagnostic_c_multiplier = c_multiplier
                                diagnostic_alt_c_weight = alt_c_weight
                                diagnostic_bc_weight = bc_weight
        diagnostic_quality = blend_quality_probabilities(
            quality_val, ordinal_val, diagnostic_ordinal_weight, q_encoder, graded_quality_order
        )
        diagnostic_blended_yield = (
            (1.0 - diagnostic_alt_c_weight) * yield_val
            + diagnostic_alt_c_weight * yield_alt_val
        )
        diagnostic_yield = adjust_yield_probabilities(
            diagnostic_blended_yield, b_val_probability,
            diagnostic_b_weight, diagnostic_c_multiplier
        )
        diagnostic_yield = blend_bc_specialist(
            diagnostic_yield, bc_val_probability, diagnostic_bc_weight
        )
        diagnostic_hierarchy = combine_hierarchical_probabilities(
            diagnostic_quality, diagnostic_yield, q_encoder, y_encoder, final_encoder
        )
        # Store the CatBoost-only path temporarily. It is replaced below by an
        # honestly cross-fitted CatBoost/ordinal quality blend.
        hierarchical_oof[val_idx] = cat_hierarchical_val
        disagreement = float(np.mean(direct_val.argmax(1) != diagnostic_hierarchy.argmax(1)))
        best_fold_blend = max(
            (
                f1_score(
                    y_final[val_idx],
                    (weight * direct_val + (1.0 - weight) * diagnostic_hierarchy).argmax(1),
                    average="macro",
                ),
                weight,
            )
            for weight in np.linspace(0.0, 1.0, 21)
        )
        component_rows.append({
            "fold": fold + 1,
            "direct_f1": direct_f1,
            "cat_quality_f1": quality_f1,
            "ordinal_quality_f1": ordinal_quality_f1,
            "yield_f1": yield_f1,
            "c_recall": c_recall,
            "b_specialist_f1": b_specialist_f1,
            "bc_specialist_f1": bc_specialist_f1,
            "drift_direct_f1": drift_direct_f1,
            "cat_hierarchical_f1": cat_hierarchical_f1,
            "ordinal_hierarchical_f1": ordinal_hierarchical_f1,
            "diagnostic_hierarchical_f1": diagnostic_hierarchical_f1,
            "diagnostic_ordinal_weight": diagnostic_ordinal_weight,
            "diagnostic_b_weight": diagnostic_b_weight,
            "diagnostic_c_multiplier": diagnostic_c_multiplier,
            "diagnostic_alt_c_weight": diagnostic_alt_c_weight,
            "diagnostic_bc_weight": diagnostic_bc_weight,
            "disagreement": disagreement,
            "diagnostic_best_blend_f1": best_fold_blend[0],
            "diagnostic_direct_weight": best_fold_blend[1],
        })
        print(
            f"  Hierarchy: cat={cat_hierarchical_f1:.4f}, ordinal={ordinal_hierarchical_f1:.4f}, "
            f"diagnostic-best={diagnostic_hierarchical_f1:.4f} "
            f"(ordinal={diagnostic_ordinal_weight:.1f}, B={diagnostic_b_weight:.2f}, "
            f"C={diagnostic_c_multiplier:.2f}, altC={diagnostic_alt_c_weight:.2f}, "
            f"BC={diagnostic_bc_weight:.2f})"
        )
        print(
            f"  Disagreement={disagreement:.3f}; diagnostic direct/hierarchy blend={best_fold_blend[0]:.4f}"
        )
        if SMOKE_STAGE == "components":
            pd.DataFrame(component_rows).to_csv(f"{OUT_DIR}/smoke_component_metrics.csv", index=False)
            print(f"[{VERSION}] Component smoke stage complete.")
            raise SystemExit(0)

        del tr_aug, val_aug, test_aug, x_tr, x_val, x_test
        del x_tr_xgb, x_val_xgb, x_test_xgb, direct_val, quality_val, quality_test
        del ordinal_val, ordinal_test, ordinal_val_cumulative, ordinal_test_cumulative
        del yield_val, yield_test, cat_hierarchical_val, ordinal_hierarchical_val
        del ordinal_quality_val, diagnostic_quality, diagnostic_blended_yield
        del diagnostic_yield, diagnostic_hierarchy
        gc.collect()

    # Cross-fit quality blending and optional B/C yield balancing before final calibration.
    quality_blend_rows = []
    hierarchical_oof.fill(0)
    hierarchical_test.fill(0)
    balanced_yield_oof = np.zeros_like(yield_oof)
    rng = np.random.default_rng(42)
    for heldout_fold in range(N_SPLITS):
        fit_mask = fold_ids != heldout_fold
        heldout_mask = fold_ids == heldout_fold
        fit_indices = np.flatnonzero(fit_mask)
        search_indices = fit_indices
        if (ENABLE_YIELD_BALANCE or ENABLE_BC_SPECIALIST) and len(fit_indices) > 400_000:
            search_indices = rng.choice(fit_indices, 400_000, replace=False)
        ordinal_grid = np.linspace(0.0, 1.0, 6) if ENABLE_YIELD_BALANCE else np.linspace(0.0, 1.0, 11)
        b_weight_grid = [0.0, 0.25, 0.5, 0.75] if ENABLE_B_SPECIALIST else [0.0]
        c_multiplier_grid = [0.8, 0.9, 1.0] if ENABLE_YIELD_BALANCE else [1.0]
        alt_c_weight_grid = np.linspace(0.0, 1.0, 6) if ENABLE_DUAL_C else [0.0]
        bc_weight_grid = np.linspace(0.0, 1.0, 5) if ENABLE_BC_SPECIALIST else [0.0]
        candidates = []
        for ordinal_weight in ordinal_grid:
            search_quality = blend_quality_probabilities(
                quality_oof[search_indices], ordinal_quality_oof[search_indices], ordinal_weight,
                q_encoder, graded_quality_order,
            )
            for alt_c_weight in alt_c_weight_grid:
                search_base_yield = (
                    (1.0 - alt_c_weight) * yield_oof[search_indices]
                    + alt_c_weight * yield_alt_oof[search_indices]
                )
                for b_weight in b_weight_grid:
                    for c_multiplier in c_multiplier_grid:
                        search_yield = adjust_yield_probabilities(
                            search_base_yield, b_specialist_oof[search_indices],
                            b_weight, c_multiplier,
                        )
                        for bc_weight in bc_weight_grid:
                            search_yield_bc = blend_bc_specialist(
                                search_yield, bc_specialist_oof[search_indices], bc_weight
                            )
                            search_hierarchy = combine_hierarchical_probabilities(
                                search_quality, search_yield_bc, q_encoder, y_encoder, final_encoder
                            )
                            candidates.append((
                                f1_score(
                                    y_final[search_indices], search_hierarchy.argmax(1), average="macro"
                                ),
                                ordinal_weight,
                                b_weight,
                                c_multiplier,
                                alt_c_weight,
                                bc_weight,
                            ))
        (
            fit_score, best_ordinal_weight, best_b_weight, best_c_multiplier,
            best_alt_c_weight, best_bc_weight,
        ) = max(
            candidates, key=lambda row: row[0]
        )
        heldout_quality = blend_quality_probabilities(
            quality_oof[heldout_mask], ordinal_quality_oof[heldout_mask], best_ordinal_weight,
            q_encoder, graded_quality_order,
        )
        heldout_base_yield = (
            (1.0 - best_alt_c_weight) * yield_oof[heldout_mask]
            + best_alt_c_weight * yield_alt_oof[heldout_mask]
        )
        heldout_yield = adjust_yield_probabilities(
            heldout_base_yield, b_specialist_oof[heldout_mask],
            best_b_weight, best_c_multiplier,
        )
        heldout_yield = blend_bc_specialist(
            heldout_yield, bc_specialist_oof[heldout_mask], best_bc_weight
        )
        balanced_yield_oof[heldout_mask] = heldout_yield
        hierarchical_oof[heldout_mask] = combine_hierarchical_probabilities(
            heldout_quality, heldout_yield, q_encoder, y_encoder, final_encoder
        )
        test_quality = blend_quality_probabilities(
            quality_test_average, ordinal_quality_test, best_ordinal_weight,
            q_encoder, graded_quality_order,
        )
        test_base_yield = (
            (1.0 - best_alt_c_weight) * yield_test_average
            + best_alt_c_weight * yield_alt_test_average
        )
        test_yield = adjust_yield_probabilities(
            test_base_yield, b_specialist_test_average,
            best_b_weight, best_c_multiplier,
        )
        test_yield = blend_bc_specialist(
            test_yield, bc_specialist_test_average, best_bc_weight
        )
        hierarchical_test += combine_hierarchical_probabilities(
            test_quality, test_yield, q_encoder, y_encoder, final_encoder
        ) / N_SPLITS
        heldout_score = f1_score(
            y_final[heldout_mask], hierarchical_oof[heldout_mask].argmax(1), average="macro"
        )
        quality_blend_rows.append({
            "heldout_fold": heldout_fold + 1,
            "ordinal_weight": best_ordinal_weight,
            "b_specialist_weight": best_b_weight,
            "c_multiplier": best_c_multiplier,
            "alt_c_weight": best_alt_c_weight,
            "bc_specialist_weight": best_bc_weight,
            "fit_f1": fit_score,
            "heldout_f1": heldout_score,
        })

    # Cross-fit every calibration and final blend choice. A zero hierarchical weight is
    # available, so an unhelpful component is never forced into the fitted blend.
    calibrated_oof = np.zeros_like(direct_oof)
    calibrated_test = np.zeros_like(direct_test)
    calibration_rows = []
    scale_rows = []
    for heldout_fold in range(N_SPLITS):
        fit_mask = fold_ids != heldout_fold
        heldout_mask = fold_ids == heldout_fold
        fit_resilient = drift_direct_oof[fit_mask] if ENABLE_DRIFT_RESISTANT else None
        heldout_resilient = drift_direct_oof[heldout_mask] if ENABLE_DRIFT_RESISTANT else None
        test_resilient = drift_direct_test if ENABLE_DRIFT_RESISTANT else None
        calibrator = fit_calibrator(
            y_final[fit_mask], direct_oof[fit_mask], hierarchical_oof[fit_mask], fit_resilient
        )
        heldout_probability = apply_calibrator(
            direct_oof[heldout_mask], hierarchical_oof[heldout_mask], calibrator, heldout_resilient
        )
        test_probability = apply_calibrator(
            direct_test, hierarchical_test, calibrator, test_resilient
        )
        regional_beta = 0.0
        if ENABLE_REGIONAL_CALIBRATION:
            inner_validation_fold = (heldout_fold + 1) % N_SPLITS
            inner_validation_mask = fold_ids == inner_validation_fold
            inner_source_mask = fit_mask & ~inner_validation_mask
            inner_resilient = drift_direct_oof[inner_source_mask] if ENABLE_DRIFT_RESISTANT else None
            inner_calibrator = fit_calibrator(
                y_final[inner_source_mask],
                direct_oof[inner_source_mask],
                hierarchical_oof[inner_source_mask],
                inner_resilient,
            )
            inner_validation_resilient = (
                drift_direct_oof[inner_validation_mask] if ENABLE_DRIFT_RESISTANT else None
            )
            inner_probability = apply_calibrator(
                direct_oof[inner_validation_mask],
                hierarchical_oof[inner_validation_mask],
                inner_calibrator,
                inner_validation_resilient,
            )
            inner_ratios, _ = fit_regional_priors(
                y_final[inner_source_mask],
                train.loc[inner_source_mask, "sigungu"],
                n_classes,
            )
            regional_candidates = []
            for beta in [0.0, 0.1, 0.2, 0.3]:
                candidate = apply_regional_priors(
                    inner_probability,
                    train.loc[inner_validation_mask, "sigungu"],
                    inner_ratios,
                    beta,
                )
                regional_candidates.append((
                    f1_score(y_final[inner_validation_mask], candidate.argmax(1), average="macro"),
                    beta,
                ))
            _, regional_beta = max(regional_candidates, key=lambda row: row[0])
            full_ratios, _ = fit_regional_priors(
                y_final[fit_mask], train.loc[fit_mask, "sigungu"], n_classes
            )
            heldout_probability = apply_regional_priors(
                heldout_probability, train.loc[heldout_mask, "sigungu"], full_ratios, regional_beta
            )
            test_probability = apply_regional_priors(
                test_probability, test_base["sigungu"], full_ratios, regional_beta
            )
        calibrated_oof[heldout_mask] = heldout_probability
        calibrated_test += test_probability / N_SPLITS
        heldout_score = f1_score(y_final[heldout_mask], calibrated_oof[heldout_mask].argmax(1), average="macro")
        calibration_rows.append({
            "heldout_fold": heldout_fold + 1,
            "mode": calibrator["mode"],
            "direct_weight": calibrator["direct_weight"],
            "resilient_weight": calibrator.get("resilient_weight", 0.0),
            "resilient_mode": calibrator.get("resilient_mode", "arithmetic"),
            "regional_beta": regional_beta,
            "fit_f1": calibrator["fit_score"],
            "heldout_f1": heldout_score,
        })
        for class_index, class_name in enumerate(final_encoder.classes_):
            scale_rows.append({
                "heldout_fold": heldout_fold + 1,
                "class": class_name,
                "direct_scale": calibrator["direct_scales"][class_index],
                "hierarchical_scale": calibrator["hierarchical_scales"][class_index],
                "resilient_scale": calibrator["resilient_scales"][class_index],
            })

    raw_direct_f1 = f1_score(y_final, direct_oof.argmax(1), average="macro")
    raw_hierarchical_f1 = f1_score(y_final, hierarchical_oof.argmax(1), average="macro")
    honest_f1 = f1_score(y_final, calibrated_oof.argmax(1), average="macro")
    print(f"\n[{VERSION}] Raw direct OOF: {raw_direct_f1:.4f}")
    print(f"[{VERSION}] Raw hierarchical OOF: {raw_hierarchical_f1:.4f}")
    print(f"[{VERSION}] Cross-fitted calibrated OOF: {honest_f1:.4f}")

    pd.DataFrame(component_rows).to_csv(f"{OUT_DIR}/component_metrics.csv", index=False)
    if proxy_metric_rows:
        pd.DataFrame(proxy_metric_rows).to_csv(f"{OUT_DIR}/proxy_metrics.csv", index=False)
    pd.DataFrame(quality_blend_rows).to_csv(f"{OUT_DIR}/quality_blend_folds.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(f"{OUT_DIR}/calibration_folds.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(f"{OUT_DIR}/class_scales.csv", index=False)
    save_class_metrics(y_final, calibrated_oof, final_encoder.classes_, f"{OUT_DIR}/oof_class_metrics.csv")
    yield_metrics_probability = (
        balanced_yield_oof
        if (ENABLE_YIELD_BALANCE or ENABLE_DUAL_C or ENABLE_BC_SPECIALIST)
        else yield_oof
    )
    save_class_metrics(
        y_yield[is_graded], yield_metrics_probability[is_graded], y_encoder.classes_,
        f"{OUT_DIR}/yield_oof_class_metrics.csv",
    )
    pd.DataFrame({"feature": lgb_features, "importance": importance}).sort_values("importance", ascending=False).to_csv(
        f"{OUT_DIR}/lgb_feature_importance.csv", index=False
    )
    if SAVE_OOF:
        np.savez_compressed(
            f"{OUT_DIR}/oof_predictions.npz",
            y_true=y_final.astype("int8"),
            fold_id=fold_ids,
            direct=direct_oof.astype("float16"),
            hierarchical=hierarchical_oof.astype("float16"),
            quality=quality_oof.astype("float16"),
            ordinal_quality=ordinal_quality_oof.astype("float16"),
            yield_probability=yield_oof.astype("float16"),
            yield_alt_probability=yield_alt_oof.astype("float16"),
            balanced_yield_probability=balanced_yield_oof.astype("float16"),
            b_specialist=b_specialist_oof.astype("float16"),
            bc_specialist=bc_specialist_oof.astype("float16"),
            drift_direct=drift_direct_oof.astype("float16"),
            calibrated=calibrated_oof.astype("float16"),
        )

    np.savez_compressed(
        f"{OUT_DIR}/test_components.npz",
        direct=direct_test.astype("float16"),
        quality=quality_test_average.astype("float16"),
        ordinal_quality=ordinal_quality_test.astype("float16"),
        yield_probability=yield_test_average.astype("float16"),
        yield_alt_probability=yield_alt_test_average.astype("float16"),
        b_specialist=b_specialist_test_average.astype("float16"),
        bc_specialist=bc_specialist_test_average.astype("float16"),
        drift_direct=drift_direct_test.astype("float16"),
        hierarchical=hierarchical_test.astype("float16"),
        calibrated=calibrated_test.astype("float16"),
    )

    test_raw["LAST_GRADE"] = final_encoder.inverse_transform(calibrated_test.argmax(1))
    output_path = f"{OUT_DIR}/260418.csv"
    test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{VERSION}] Saved {output_path}; total time {elapsed(started)}")


if __name__ == "__main__":
    main()
