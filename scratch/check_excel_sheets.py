import openpyxl
import os

data_dir = "C:/Users/jaeyo/Projects/Animals/data"
kpn_path = os.path.join(data_dir, "KPN 유전능력 자료.xlsx")

wb = openpyxl.load_workbook(kpn_path, read_only=True)
print("Sheet names in Excel file:")
print(wb.sheetnames)
