from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# グローバル変数
FILE_HEADER_SIZE = 14

def readBitMapData(src:str) -> tuple[dict, bytearray,np.ndarray]:
    """
        - Description
            BitMap形式の画像データを読み込む

        - Arg
            src:str             -> 読み込む画像のパス

        - Ret
            header:bytearray    -> 画像のヘッダー情報
            img:np.ndarray      -> 画像のバイナリデータ
    """

    dat = bytearray()
    with open(src, "rb") as f:
        dat += f.read()

    file_header = getBitMapHeader(dat=dat)
    header = dat[:file_header["bfOffBits"]]
    img = np.array([x for x in dat[file_header["bfOffBits"]:]])
    img = img.reshape(int(img.shape[0]/3), 3)

    return (file_header, header, img)

def getBitMapHeader(dat:bytearray) -> dict:
    """
        - Description
            BitMap画像のヘッダ情報を取得(今回はファイルヘッダまで)
        - Arg
            header:bytearray    -> ヘッダのバイナリ配列
        - Ret
            info:dict           -> ヘッダ情報
    """
    info = {}
    info["bfType"] = bytearray.decode(dat[0:2])
    info["bfSize"] = int.from_bytes(dat[2:6], "little")
    info["bfReserved1"] = int.from_bytes(dat[6:8], "little")
    info["bfReserved2"] = int.from_bytes(dat[8:10], "little")
    info["bfOffBits"] = int.from_bytes(dat[10:14], "little")

    return info

def getGrayScale(img:np.ndarray) -> np.ndarray:
    """
        - Description
            グレースケールの画像を取得する
        - Arg
            img:np.ndarray  -> 元画像
        - Ret
            gray_img:np.ndarray -> グレースケール画像
    """
    gray_img = np.mean(img, axis=1)
    return gray_img

def saveHist(img:np.ndarray, output_path:str):
    """
        - Description
            入力されたグレースケール画像のヒストグラムを画像として保存する
        - Arg
            img:np.ndarray  -> グレースケール画像
            output_path:str -> 保存先パス
        - Ret:
            None
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(img, range=(0, 255), bins=256)
    plt.savefig(output_path)

def convert_LGLT(img:np.ndarray) -> np.ndarray:
    """
        - Description
            濃度線形変換を行う
        - Arg
            img:np.ndarray      -> 入力画像（グレースケール）
        - Ret
            ret_img             -> 濃度線形変換後の画像
    """
    ret_img = img - np.min(img)  # 最小値を0に
    saveHist(img=ret_img, output_path="test1.jpg")
    ret_img = ret_img * (255 / (np.max(img)-np.min(img)))  # 最大値が255になるように乗算
    saveHist(img=ret_img, output_path="test2.jpg")
    return ret_img

def saveBitMap(img:np.ndarray, header:bytearray, output_path:str):
    """
        - Description
            BitMap形式で保存を行う
        - Arg
            img:np.ndarray      -> 保存する画像
            header:bytearray    -> ヘッダ情報
        - Ret
            None
    """
    dat = header + img.astype(np.uint8).tobytes()
    with open(output_path, "wb") as f:
        f.write(dat)


if __name__ == "__main__":
    src_path = Path("./data/Lenna.bmp")
    gray_path = Path("./data/Lenna_gray.bmp")
    hist_path = Path("./data/Lenna_hist.jpg")

    info, header, img = readBitMapData(str(src_path))
    gray_img = getGrayScale(img=img)
    gray_img = convert_LGLT(img=gray_img)
    gray_save_img = np.zeros((gray_img.shape[0],3))
    for i in range(3):
        gray_save_img[:,i] = gray_img
    saveBitMap(img=gray_save_img, header=header, output_path=str(gray_path))

