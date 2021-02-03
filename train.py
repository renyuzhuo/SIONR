import torch
from argparse import ArgumentParser
import pandas as pd
from SIONR import SIONR
from dataset import VideoFeatureDataset
import numpy as np
from fit_function import fit_function
from scipy import stats
import torch.optim as optim

gpu_device = 'cuda:0'
train_batch_size = 1
num_workers = 3
lr = 1e-3
database = 'KoNViD-1k'

video_dir = '/content/SIONR/KoNViD_1k_videos'
feature_dir = '/content/drive/MyDrive/KoNViD/data/CNN_features_KoNViD-1k/'
info = pd.read_csv('/content/SIONR/data/KoNViD_1k_attributes.csv')
file_names = info['flickr_id'].values
video_name = [str(k) + '.mp4' for k in file_names]
video_name = np.array(video_name)
mos = info['MOS'].values
database_info = {'video_dir': video_dir,
                  'feature_dir': feature_dir,
                  'video_name': video_name,
                  'mos': mos}
scale = mos.max()

split_idx_file = 'data/train_val_test_split.xlsx'
split_info = pd.read_excel(split_idx_file)
idx_all = split_info.iloc[:, 0].values
split_status = split_info['status'].values
train_idx = idx_all[split_status == 'train']

device = torch.device(gpu_device)

#加载自己的模型，并非自带部分预训练模型，部分预训练模型code可以用于cgn
VQA_model = SIONR()
model_file = 'model/SIONR.pt'
VQA_model.load_state_dict(torch.load(model_file))
VQA_model = VQA_model.to(device)

train_dataset = VideoFeatureDataset(idx_list=train_idx, database_info=database_info)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                                          num_workers=num_workers)


SIONR_params = list(map(id, VQA_model.parameters()))
#base_params = filter(lambda p: id(p) not in SIONR_params, model.parameters())
optimizer = optim.Adam([{'params': VQA_model.parameters(), 'lr': 1e-3}], lr=lr)



# train
y_predict = np.zeros(len(train_idx))
y_label = np.zeros(len(train_idx))

loss = [0 for _ in range(1)]
loss_sum = []
for i, data in enumerate(train_loader):
    video = data['video'].to(device)
    feature = data['feature'].to(device)
    label = data['score']
    label = label.to(device)
    y_label[i] = label.item()
    #
    optimizer.zero_grad()
    outputs, batch_info = VQA_model(video, feature, label, True)
    batch_info.backward()
    optimizer.step()
    #

    y_predict[i] = outputs.item()

    print('train: ', i)

y_predict = fit_function(y_predict, y_label)
train_PLCC = stats.pearsonr(y_predict, y_label)[0]
train_SROCC = stats.spearmanr(y_predict, y_label)[0]
train_RMSE = np.sqrt((((y_predict - y_label) * scale) ** 2).mean())

result_excel = 'result/train_result.xlsx'
result = pd.DataFrame()
result['PLCC'] = [train_PLCC]
result['SROCC'] = [train_SROCC]
result['RMSE'] = [train_RMSE]
result.to_excel(result_excel)
