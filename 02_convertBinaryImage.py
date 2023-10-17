from pathlib import Path
import numpy as np
import cv2

# グローバル変数
FILE_HEADER_SIZE = 14
THRESH_OTSU = "otsu"

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

def convertBinaryData(img:np.ndarray, method:str=THRESH_OTSU) -> np.ndarray:
    """
        - Description
            二値化を行う

        - Arg
            img:bytearray       -> 二値化前の画像(グレースケール)

        - Ret
            bin_img:bytearray   -> 二値化後の画像
    """
    bin_img = img
    if method == THRESH_OTSU:
        inter_vals = list()
        for th in range(0, 256):
            inter_val = 0
            if np.count_nonzero(img<th) > 0 and np.count_nonzero(img>=th) > 0:
                mean0 = np.mean(img)
                mean1 = np.mean(img[img<th])
                count1 = np.count_nonzero(img<th)
                mean2 = np.mean(img[img>=th])
                count2 = np.count_nonzero(img>=th)
                inter_val = count1*((mean1-mean0)**2) + count2*((mean2-mean0)**2) / (count1 + count2)
            inter_vals.append(inter_val)
        threshold = np.nanargmax(inter_vals)
        bin_img = np.where(img<threshold, 0, 255)
    else:
        print("unknown method")

    return bin_img

def saveBitMap(img:np.ndarray, header:bytearray):
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
    with open(str(output_path), "wb") as f:
        f.write(dat)


if __name__ == "__main__":
    src_path = Path("./data/Lenna.bmp")
    output_path = Path("./data/Lenna_output_binary.bmp")

    info, header, img = readBitMapData(str(src_path))
    bin_img = convertBinaryData(img=img[:,0], method=THRESH_OTSU)
    img[:,0], img[:,1], img[:,2]= bin_img, bin_img, bin_img

    saveBitMap(img=img, header=header)
