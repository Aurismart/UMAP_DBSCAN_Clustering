import cuml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from tqdm import tqdm
# from cuml.cluster import DBSCAN
from cuml.cluster import HDBSCAN
import shutil
import datetime

'''
將PATH更換成資料集的位置,
程式碼於WSL運行,
已將WINDOWS硬碟掛載到/mnt,
因此/mnt/d 這個路徑對應到 windows的 D:/
'''
PATH = '/mnt/d/umap_data/231003_UMAP_3F_Transition'


def log_transform_and_normalize(x):
    '''
    對頻譜圖進行對數轉換的函數
    '''
    x = np.maximum(x, 0)
    x_log1p = np.log1p(x)
    return (x_log1p - np.min(x_log1p)) / (np.max(x_log1p) - np.min(x_log1p))


'''
載入檔案的部分
'''
print("start loading data!")
if not os.path.exists(os.path.join(PATH,os.path.split(PATH)[1]+".npz")):
    '''
    若找不到暫存檔則開始轉換音檔為頻譜圖
    '''
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
    '''
    若找到暫存檔則直接載入
    '''
    npz_temp = np.load(os.path.join(PATH,os.path.split(PATH)[1]+".npz"))  
    mels_2d,fn_order = npz_temp["data_2d"],npz_temp["fn_order"]
print("loading finished!")



'''
10/03 使用UMAP+DBSCAN找到一組超參數為 n_neighbors=7,spread=1.5,eps=0.8

n_neighbors:大小影響降維時保留特徵的傾向,數值越大越傾向保留共同的特徵,數值越小則傾向保留局部特徵
主要表現在小的n_neighbors會讓相似的點更密集,但是造成跟稍有差異的資料離得更遠,數值大則相反

spread:數值越大資料越分散,但是數值太大會導致分群困難

eps:決定多少兩個點的距離小於多少才會被認為在同一捆

min_samples:每個資料核心點附近最少要多少筆資料才會形成一群


10/03 更換DBSCAN算法,改用HDBSCAN

'''


'''
設定要搜尋的超參數
可使用range 或是直接輸入到列表
range 用法為 range(start,end,step)

range(4,12) 等價於 [4,5,6,7,8,9,10,11]
np.arange(0.5,1.2,0.1) 等價於 [0.5,0.6,0.7,0.8,0.9,1.0,1.1]

若步長為浮點數則需要使用np.arange
'''
# n_neighbors_range = range(4,12)
# spread_range = np.arange(1.0,1.6,0.5)
# eps_range = np.arange(0.5,1.2,0.1)
n_neighbors_range = [7]
spread_range = [1.5]
# eps_range = [0.8]

'''
產生圖框
'''
plt.figure(figsize=(10, 8))


'''
為了畫進度條,先把任務長度算出來
'''
total_iterations = len(n_neighbors_range) * len(spread_range)


'''
使用tqdm包裹任務顯示進度條
'''
with tqdm(total=total_iterations, desc="Overall progress") as pbar:

    '''
    使用迴圈進行多組參數進行UMAP訓練
    '''
    for n_neighbors in n_neighbors_range:
        # for min_d in min_dist_range:
        for spread in spread_range:
            
            # for eps in eps_range:
            # print("umap training start!")

            '''
            UMAP有哪些參數可以設定需要參照NVIDIA RAPIDS cuml文件
            文件網址: https://docs.rapids.ai/api/cuml/stable/api/
            
            fit:只進行訓練umap model
            fit_transform:進行訓練之後,將訓練資料降維的結果回傳
            transform:使用現在的模型參數將資料降維回傳
            '''
            umap_model = cuml.UMAP(n_neighbors=n_neighbors,spread = spread, init="spectral")
            embedding = umap_model.fit_transform(mels_2d)
            # print("umap training done!")

            # print("DBSCAN training start!")
            # db = DBSCAN(eps=eps, min_samples=75)
            # labels = db.fit_predict(embedding)

            '''
            使用HDNSCAN進行分群,參數設定一樣參照上述NVIDIA RAPIDS cuml文件
            fit:只進行訓練hdbscan model
            fit_predict:進行訓練之後,將訓練資料分群的結果回傳
            '''
            hdbscan = HDBSCAN(min_cluster_size=1000,min_samples = 1000)
            labels = hdbscan.fit_predict(embedding)

            # print("DBSCAN training done!")
            

            '''
            將座標點根據labels(HDBSCAN產生的結果)標在圖片上
            '''
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
            

            '''
            選0.01比例的資料,將這些資料標上時間
            '''
            num_to_label = int(0.01* len(fn_order))
            selected_indices = np.random.choice(len(fn_order), num_to_label, replace=False)
            for i in selected_indices:
                
                '''
                從檔名中取出時間
                '''
                text = fn_order[i][fn_order[i].index("_")+1:fn_order[i].rindex("_")]
                text = text[0:5]
                
                '''
                把文字按點標上去
                '''
                plt.annotate(text, (embedding[i, 0], embedding[i, 1]))


            '''
            用現在的時間產生字串,再使用這個字串產生資料夾避免檔案被覆寫
            '''
            now = datetime.datetime.now()
            formatted_date  = now.strftime('%m%d_%H')
            save_root = '/mnt/c/Users/znhea/OneDrive/桌面/Umap'
            save_root = os.path.join(save_root,formatted_date)
            
            '''
            若沒有目標路徑則創建資料夾
            '''
            if not os.path.isdir(save_root):
                os.mkdir(save_root)


            '''
            幫圖片上標題,存檔,後把記憶體的圖片內容清除
            '''
            plt.title('UMAP/HDBSCAN Clustering')
            plt.savefig(os.path.join(save_root,f"UMAP_DBSCAN_Clustering_n_neighbors{n_neighbors}_spread{spread}.png"))
            plt.cla()
            

            '''
            根據分群的結果將音檔複製到指定路徑,調參時把下面的迴圈註解,直到調到一個滿意的參數才啟動
            '''
            for i in tqdm(range(len(labels))):
                '''
                若沒有目標路徑則創建資料夾
                '''
                if not os.path.isdir(save_root):
                    os.mkdir(save_root)

                '''
                HDBSCAN將雜音分到-1
                '''
                if labels[i] == -1:
                    target_root = os.path.join(save_root,"noise")
                else:
                    target_root = os.path.join(save_root,str(labels[i]))

                '''
                若沒有目標路徑則創建資料夾
                '''
                if not os.path.isdir(target_root):
                    os.mkdir(target_root)

                '''
                複製檔案
                '''
                shutil.copy2(os.path.join(PATH,fn_order[i]),os.path.join(target_root,fn_order[i]))

            '''
            更新進度條
            '''
            pbar.update(1)