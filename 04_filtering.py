import cv2
import numpy as np

### グローバル変数
# ラプラシアンフィルタ
LAPLACIAN_FILTER = np.array([[ 0,  1,  0],
                             [ 1, -4,  1],
                             [ 0,  1,  0]])
# 右シフトフィルタ
RIGHT_SHIFT_FILTER = np.array([[ 0,  0,  0],
                               [ 1,  0,  0],
                               [ 0,  0,  0]])
# オリジナルフィルタ
ORIG_FILTER = np.array([[ 0,  1,  0],
                        [ 2, -4,  0],
                        [ 0,  1,  0]])


def convolution(img:np.ndarray, filter:np.ndarray, offset:int) -> np.ndarray:
    """
        Description:
            畳み込み積分を行う
        Arg:
            img:np.ndarray -> 入力画像（グレースケール）
            filter:np.ndarray -> 適用フィルタ
        Retern:
            フィルタリング後の画像
    """
    ysize, xsize = img.shape  # 画像サイズ取得
    output = np.zeros((ysize, xsize))  # 元のサイズと同じサイズで0(黒)埋めした画像用意
    for y in range(1, ysize-1):
        for x in range(1, xsize-1):
            for j in range(-1, 2):
                for i in range(-1, 2):
                    output[y,x] += filter[j+1][i+1] * img[y+j][x+i]  # 畳み込み
    output += offset
    return output

if __name__ == "__main__":
    img = cv2.imread("./data/Lenna.bmp")  # 画像の入力
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール化
    edge_img = convolution(img=gray_img, filter=LAPLACIAN_FILTER, offset=0)  # エッジ計算
    # 右シフト計算
    right_img = convolution(img=gray_img, filter=RIGHT_SHIFT_FILTER, offset=0)
    for i in range(100-1):
        right_img = convolution(img=right_img, filter=RIGHT_SHIFT_FILTER, offset=0)
    orig_img = convolution(img=gray_img, filter=ORIG_FILTER, offset=128)  # オリジナルフィルタ適用

    # 出力保存
    cv2.imwrite("./data/Lenna_gray.jpg", gray_img)
    cv2.imwrite("./data/Lenna_edge.jpg", edge_img)
    cv2.imwrite("./data/Lenna_right.jpg", right_img)
    cv2.imwrite("./data/Lenna_orig.jpg", orig_img)