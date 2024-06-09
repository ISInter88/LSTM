import pandas as pd

def write_last_20_percent(input_csv, output_csv):
    # CSVファイルを読み込む
    df = pd.read_csv(input_csv)
    
    # 全体の行数を取得
    total_rows = len(df)
    
    # 2割の行数を計算
    start_row = int(total_rows * 0.8)
    
    # 2列目から20%を超えた部分のみを取得
    subset = df.iloc[start_row:, :]  # ilocで行と列を指定、行はstart_rowから最後まで、列は2列目（indexは1）
    
    # 新しいDataFrameを作成（単一列のDataFrame）
    result_df = pd.DataFrame(subset)
    
    # 結果を新しいCSVファイルに書き出す
    result_df.to_csv(output_csv, index=False, header=False)

# 使用例
input_csv = 'datasets/Abilene_30.csv'  # 読み込むCSVファイルのパス
output_csv = 'Abilene_test_part.csv'  # 書き出すCSVファイルのパス

write_last_20_percent(input_csv, output_csv)