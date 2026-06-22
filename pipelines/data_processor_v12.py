"""V12 processor: expose test-safe extended pedigree identifiers."""

import os

import pandas as pd

from data_processor_v8 import HanwooDataProcessorV8


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
