import cuml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from tqdm import tqdm
from cuml.cluster import DBSCAN
from cuml.cluster import HDBSCAN
import shutil
import datetime

PATH = '/mnt/d/umap_data/231003_UMAP_3F_Transition'

def log_transform_and_normalize(x):
    x = np.maximum(x, 0)
    x_log1p = np.log1p(x)
    return (x_log1p - np.min(x_log1p)) / (np.max(x_log1p) - np.min(x_log1p))

print("start loading data!")
if not os.path.exists(os.path.join(PATH,os.path.split(PATH)[1]+".npz")):
    mels = []
    fn_order = []
    for file_name in tqdm(os.listdir(PATH)):
        if not file_name.endswith(".wav"):
            continue
        audio, sr = librosa.load(os.path.join(PATH,file_name),sr=16000,res_type='kaiser_fast')
        data = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft = 1024,hop_length = 256 ,n_mels = 64)
        data = log_transform_and_normalize(data)
        mels.append(data)
        # fn_order.append(file_name[file_name.index("_")+1:file_name.rindex("_")])
        fn_order.append(file_name)
    mels_3d = np.array(mels)
    mels_2d = mels_3d.reshape(mels_3d.shape[0], -1)
    np.savez(os.path.join(PATH,os.path.split(PATH)[1]+".npz"),data_2d = mels_2d,fn_order = fn_order)
else:
    npz_temp = np.load(os.path.join(PATH,os.path.split(PATH)[1]+".npz"))  
    mels_2d,fn_order = npz_temp["data_2d"],npz_temp["fn_order"]
print("loading finished!")
# n_neighbors_range = range(4,12)
# spread_range = np.arange(1.0,1.6,0.5)
# eps_range = np.arange(0.5,1.2,0.1)


'''
10/03 使用UMAP+DBSCAN找到一組超參數為 n_neighbors=7,spread=1.5,eps=0.8

n_neighbors:大小影響降維時保留特徵的傾向,數值越大越傾向保留共同的特徵,數值越小則傾向保留局部特徵
主要表現在小的n_neighbors會讓相似的點更密集,但是造成跟稍有差異的資料離得更遠,數值大則相反

spread:數值越大資料越分散,但是數值太大會導致分群困難

eps:決定多少兩個點的距離小於多少才會被認為在同一捆

min_samples:每個資料核心點附近最少要多少筆資料才會形成一群


10/03 更換DBSCAN算法,改用HDBSCAN

'''
n_neighbors_range = [7]
spread_range = [1.5]
eps_range = [0.8]

plt.figure(figsize=(10, 8))
total_iterations = len(n_neighbors_range) * len(spread_range) * len(eps_range)
with tqdm(total=total_iterations, desc="Overall progress") as pbar:
    for n_neighbors in n_neighbors_range:
        # for min_d in min_dist_range:
        for spread in spread_range:
            # for eps in eps_range:
            # print("umap training start!")
            # umap_model = cuml.UMAP(n_neighbors=n_neighbors,min_dist = min_d, init="spectral")
            umap_model = cuml.UMAP(n_neighbors=n_neighbors,spread = spread, init="spectral")
            embedding = umap_model.fit_transform(mels_2d)
            # print("umap training done!")

            # print("DBSCAN training start!")
            # db = DBSCAN(eps=eps, min_samples=75)
            # labels = db.fit_predict(embedding)

            hdbscan = HDBSCAN(min_cluster_size=1000,min_samples = 1000)
            labels = hdbscan.fit_predict(embedding)
            
            # print("DBSCAN training done!")

            # print("Isolation Forest training start!")
            # iforest = IsolationForest(contamination=0.05)  # 假设5%的数据点是异常的
            # outliers = iforest.fit_predict(embedding)
            # print("Isolation Forest training done!")

            num_to_label = int(0.01* len(fn_order))
            selected_indices = np.random.choice(len(fn_order), num_to_label, replace=False)


            # print("start visualize the result!")
            # Visualizing the clustering result
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
            # scatter = plt.plot(embedding[:, 0], embedding[:, 1],marker = "o")
            # print("scatter done!")

            # print("start annotate only the selected points!")
            # Annotate only the selected points
            for i in selected_indices:
                text = fn_order[i][fn_order[i].index("_")+1:fn_order[i].rindex("_")]
                text = text[0:5]
                plt.annotate(text, (embedding[i, 0], embedding[i, 1]))
            # print("visualize done!")


            now = datetime.datetime.now()
            formatted_date  = now.strftime('%m%d_%H')
            save_root = '/mnt/c/Users/znhea/OneDrive/桌面/Umap'
            save_root = os.path.join(save_root,formatted_date)

            if not os.path.isdir(save_root):
                os.mkdir(save_root)
            plt.title('UMAP/HDBSCAN Clustering')
            # plt.colorbar(scatter)
            # plt.show()

            # plt.savefig(os.path.join(save_root,f"UMAP_DBSCAN_Clustering_n_neighbors{n_neighbors}_min_dist{min_d}.png"))
            plt.savefig(os.path.join(save_root,f"UMAP_DBSCAN_Clustering_n_neighbors{n_neighbors}_spread{spread}.png"))
            plt.cla()
            

            for i in tqdm(range(len(labels))):

                if not os.path.isdir(save_root):
                    os.mkdir(save_root)

                if labels[i] == -1:
                    target_root = os.path.join(save_root,"noise")
                else:
                    target_root = os.path.join(save_root,str(labels[i]))

                if not os.path.isdir(target_root):
                    os.mkdir(target_root)

                shutil.copy2(os.path.join(PATH,fn_order[i]),os.path.join(target_root,fn_order[i]))
                # print(f'lable = {labels[i]}')
                # print(f'file name = {fn_order[i]}')d
            pbar.update(1)  # 更新进度条