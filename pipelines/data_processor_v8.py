import ast
import os
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


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
