from pathlib import Path
import cv2

src_path = Path("./data/Lenna.bmp")
output_path = Path("./data/Lenna_output.bmp")

### Data reading
# ①BMPファイルの内容を読み込み変数に代入する
img = cv2.imread(str(src_path))

### Image processing

### Dat writing
# ②変数に代入されたデータをBMPファイルとして書き出す
cv2.imwrite(str(output_path), img)