import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
skeleton_arr = np.array(skeletons)

num_clusters = 4
x = torch.from_numpy(skeleton_arr)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

np.random.seed(123)

cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=device
)
print(cluster_ids_x)
idx = cluster_ids_x.numpy()
np.save("/content/drive/MyDrive/output_numpy_file_new", idx)
