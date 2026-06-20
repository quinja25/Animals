import os

import numpy as np
import pandas as pd


class HanwooDataProcessorV7:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.kpn_bv = None
        self.lineage = None
        self.weather_monthly = None
        self.weather_daily = None
        self.farm_profile = None

        self.QUALITY_ORDER = ["1++", "1+", "1", "2", "3", "등외"]
        self.YIELD_ORDER = ["A", "B", "C"]
        self.FINAL_GRADES = [
            "1++A", "1++B", "1++C",
            "1+A", "1+B", "1+C",
            "1A", "1B", "1C",
            "2A", "2B", "2C",
            "3A", "3B", "3C",
            "등외",
        ]
        self.GRADE_SCORE = {grade: 15 - index for index, grade in enumerate(self.FINAL_GRADES)}

    def load_auxiliary_data(self):
        print("[DataProcessorV7] Loading auxiliary data...")

        kpn_path = os.path.join(self.data_dir, "KPN 전산입력 자료.xlsx")
        if os.path.exists(kpn_path):
            try:
                kpn_df = pd.read_excel(kpn_path)
                rename_map = {
                    "KPN번호": "KPN_NO",
                    "체중선발지수육종가": "kpn_weight_sbv",
                    "쇄심형지수육종가": "kpn_rea_sbv",
                    "쇄기지수육종가": "kpn_backfat_sbv",
                    "근내지방도 육종가": "kpn_insfat_sbv",
                }
                available = [col for col in rename_map if col in kpn_df.columns]
                if available:
                    if "KPN번호" in kpn_df.columns:
                        kpn_df["KPN번호"] = kpn_df["KPN번호"].astype(str).str.strip()
                    self.kpn_bv = kpn_df[available].rename(columns={col: rename_map[col] for col in available})
                else:
                    self.kpn_bv = None
            except Exception as exc:
                print(f"[DataProcessorV7] Skipping KPN Excel file: {exc}")
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
            for window in [6, 12]:
                wm[f"s_ta_{window}m"] = wm.groupby("stn")["s_ta"].transform(
                    lambda series: series.rolling(window, min_periods=1).mean()
                )
                wm[f"s_thi_{window}m"] = wm.groupby("stn")["s_thi"].transform(
                    lambda series: series.rolling(window, min_periods=1).mean()
                )
                wm[f"s_heat_{window}m"] = wm.groupby("stn")["s_heat_days"].transform(
                    lambda series: series.rolling(window, min_periods=1).sum()
                )

            self.weather_monthly = wm.astype({"stn": int, "year": int, "month": int}).astype("float32", errors="ignore")
            self.weather_monthly[["stn", "year", "month"]] = self.weather_monthly[["stn", "year", "month"]].astype(int)

        area_path = os.path.join(self.data_dir, "hanwoo_area.csv")
        death_path = os.path.join(self.data_dir, "hanwoo_death.csv")
        if os.path.exists(area_path):
            area = pd.read_csv(area_path).drop_duplicates("FARM_UNIQUE_NO")
            if os.path.exists(death_path):
                death = pd.read_csv(death_path)
                death_cnt = death.groupby("FARM_UNIQUE_NO").size().reset_index(name="death_cnt")
                area = area.merge(death_cnt, on="FARM_UNIQUE_NO", how="left").fillna(0)
            else:
                area["death_cnt"] = 0

            area["density"] = (area["AREA"] / (area["C2023"] + 1)).astype("float32")
            area["farm_size"] = pd.cut(
                area["C2023"],
                bins=[-1, 30, 100, 100000],
                labels=[0, 1, 2],
            ).cat.add_categories([-1]).fillna(-1).astype(int)
            self.farm_profile = area[["FARM_UNIQUE_NO", "density", "farm_size", "death_cnt"]].copy()
            self.farm_profile["FARM_UNIQUE_NO"] = self.farm_profile["FARM_UNIQUE_NO"].astype(str).str.strip()

    def fit_target_stats(self, train_df):
        print("[DataProcessorV7] Calculating target statistics...")
        df = train_df.copy()
        df["LAST_GRADE"] = df["LAST_GRADE"].fillna("등외").astype(str)
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

        self.stn_stats = (
            df.groupby("stn")["grade_score"]
            .mean()
            .reset_index()
            .rename(columns={"grade_score": "stn_grade_avg"})
        )

    def transform(self, df, is_train=True):
        print(f"[DataProcessorV7] Transforming {'train' if is_train else 'test'} data...")
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

        sex_map = {"암": 0, "수": 1, "거세": 2}
        if "JUDGE_SEX" in df.columns:
            df["sex_code"] = df["JUDGE_SEX"].map(sex_map).fillna(-1).astype(int)
        else:
            df["sex_code"] = -1
        df["weight_per_age"] = (df["WEIGHT"] / (df["AGE"] + 1)).astype("float32")

        if self.weather_monthly is not None:
            df = df.merge(
                self.weather_monthly,
                left_on=["stn", "abatt_year", "abatt_month"],
                right_on=["stn", "year", "month"],
                how="left",
            )
            df.drop(columns=["year", "month"], inplace=True)

        if self.farm_profile is not None:
            df = df.merge(self.farm_profile, on="FARM_UNIQUE_NO", how="left")

        if self.lineage is not None:
            df = df.merge(self.lineage, on="CATTLE_NO", how="left")

        if self.kpn_bv is not None and "KPN_NO" in df.columns:
            df = df.merge(self.kpn_bv, on="KPN_NO", how="left")

        if hasattr(self, "kpn_stats") and self.kpn_stats is not None:
            df = df.merge(self.kpn_stats, on="KPN_NO", how="left")
        if hasattr(self, "stn_stats") and self.stn_stats is not None:
            df = df.merge(self.stn_stats, on="stn", how="left")

        if is_train and "LAST_GRADE" in df.columns:
            df["LAST_GRADE"] = df["LAST_GRADE"].fillna("등외").astype(str)
            df["target_q"] = df["LAST_GRADE"].str.extract(r"^(1\+\+|1\+|1|2|3|등외)")[0].fillna("등외")
            df["target_y"] = df["LAST_GRADE"].str.extract(r"(A|B|C)$")[0].fillna("B")

        drop_cols = [
            "JUDGE_SEX",
            "sido",
            "sigungu",
            "eupmyeondong",
            "CATTLE_NO",
            "ABATT_DATE",
            "BIRTH_YMD",
            "JUDGE_DATE",
        ]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        return df
