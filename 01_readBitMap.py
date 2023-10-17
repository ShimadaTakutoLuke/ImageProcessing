from pathlib import Path

src_path = Path("./data/Lenna.bmp")
output_path = Path("./data/Lenna_output.bmp")

# ### Data reading
# # ①BMPファイルの内容を読み込み変数に代入する
img = bytearray()  # バイナリ配列変数の定義
with open(str(src_path), "rb") as f:  # バイナリファイルとして画像読み込み
    img += f.read()  # 読み込んだバイナリデータを変数に格納

# ### Image processing

# ### Dat writing
# # ②変数に代入されたデータをBMPファイルとして書き出す
with open(str(output_path), "wb") as f:  # バイナリファイルとして書き込み
    f.write(img)  # バイナリ配列をファイルに書き込み