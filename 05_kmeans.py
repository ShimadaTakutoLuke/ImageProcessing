import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def set_adjacent_coordinates(x, y, cluster, src, dist):
    """
    与えられた座標の値と同じ値を持つ隣り合った座標を探索し、distに値をセットする
    """
    original_cluster = src[y][x]
    stack = [(x, y)]
    check_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while stack:
        x, y = stack.pop()
        for i, j in check_list:
            if y + i < 0 or y + i >= src.shape[0] or x + j < 0 or x + j >= src.shape[1]:
                continue
            if src[y + i][x + j] == original_cluster and dist[y + i][x + j] == 0:
                dist[y + i][x + j] = cluster
                stack.append((x + j, y + i))

img = cv2.imread("./data/tsumiki.jpg")  # 画像の入力

# 画像をK-means法によって分割
print("--- kmeans start ---")
kmeans_img = KMeans(n_clusters=10).fit_predict(img.reshape(-1, 3)).reshape(img.shape[0], img.shape[1])
# 画像の各領域の境界線を描画
print("--- boundary plot start ---")
img_boundary = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(kmeans_img.shape[0]):
    for j in range(kmeans_img.shape[1]):
        if j + 1 < kmeans_img.shape[1] and kmeans_img[i][j] != kmeans_img[i][j + 1]:
            img_boundary[i][j] = 255
        if i + 1 < kmeans_img.shape[0] and kmeans_img[i][j] != kmeans_img[i + 1][j]:
            img_boundary[i][j] = 255

# 画像の重心の計算してクラスタ番号を描画
print("--- number plot start ---")
numberd_img = img_boundary.copy()
for i in range(1, 10):
    x_sum = 0
    y_sum = 0
    count = 0
    for j in range(kmeans_img.shape[0]):
        for k in range(kmeans_img.shape[1]):
            if kmeans_img[j][k] == i:
                x_sum += k
                y_sum += j
                count += 1
    x = int(x_sum / count)
    y = int(y_sum / count)
    numberd_img = cv2.putText(numberd_img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
plt.imsave("./data/tsumiki_clustered.jpg", kmeans_img)
plt.imsave("./data/tsumiki_boundary.jpg", img_boundary)
plt.imsave("./data/tsumiki_numberd.jpg", numberd_img)

# 隣り合った領域を同じクラスタにする
print("--- split start ---")
cluster = 1
img_seg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(img_seg.shape[0]):
    for j in range(img_seg.shape[1]):
        if img_seg[i][j] == 0:
            img_seg[i][j] = cluster
            set_adjacent_coordinates(j, i, cluster, kmeans_img, img_seg)
            cluster += 1

# 画像の各領域の境界線を描画
print("--- boundary plot start ---")
img_boundary = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(img_seg.shape[0]):
    for j in range(img_seg.shape[1]):
        if j + 1 < img_seg.shape[1] and img_seg[i][j] != img_seg[i][j + 1]:
            img_boundary[i][j] = 255
        if i + 1 < img_seg.shape[0] and img_seg[i][j] != img_seg[i + 1][j]:
            img_boundary[i][j] = 255

# 画像の重心の計算してクラスタ番号を描画
print("--- number plot start ---")
numberd_img = img_boundary.copy()
for i in range(1, cluster):
    x_sum = 0
    y_sum = 0
    count = 0
    if np.count_nonzero(img_seg == i) == 0:
        continue
    for j in range(img_seg.shape[0]):
        for k in range(img_seg.shape[1]):
            if img_seg[j][k] == i:
                x_sum += k
                y_sum += j
                count += 1
    x = int(x_sum / count)
    y = int(y_sum / count)
    numberd_img = cv2.putText(numberd_img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

plt.imsave("./data/tsumiki_clustered1.jpg", img_seg)
plt.imsave("./data/tsumiki_boundary1.jpg", img_boundary)
plt.imsave("./data/tsumiki_numberd1.jpg", numberd_img)