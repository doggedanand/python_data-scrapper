import pdfplumber
import pandas as pd

print("Table extraction started")

file_path = "18-Jan-Paper-I-EN.pdf"

table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",  
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 50,
}

def looks_like_real_table(table):
    if not table or len(table) < 3:   
        return False
    
    # Count how many columns have non-empty values across rows
    col_counts = [sum(1 for row in table if row and i < len(row) and row[i]) 
                  for i in range(len(table[0]))]
    
    # Require at least 3 columns with values in most rows
    dense_cols = sum(1 for c in col_counts if c >= len(table) // 2)
    return dense_cols >= 3


found_any = False  

with pdfplumber.open(file_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables(table_settings)
        for i, table in enumerate(tables, start=1):
            if looks_like_real_table(table):
                df = pd.DataFrame(table[1:], columns=table[0])
                print(f"\nPage {page_num}, Table {i} (DETECTED)")
                print(df)
                found_any = True

if not found_any:
    print("\n No valid tables were found in this PDF.")
