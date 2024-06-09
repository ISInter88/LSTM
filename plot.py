import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパスを指定
csv_file = 'Abilene_test_part.csv'

# CSVファイルをDataFrameとして読み込む
df = pd.read_csv(csv_file)

# グラフのサイズを指定
plt.figure(figsize=(10, 6))

# 7つの特徴量を順にプロット
for i in range(1,7):
    feature_name = df.columns[i]  # 特徴量の名前
    plt.subplot(3, 3, i+1)
    plt.plot(df[feature_name])
    plt.title(feature_name)

# レイアウトの調整
plt.tight_layout()

# グラフを表示
plt.show()