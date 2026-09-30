import numpy as np
import csv
from implementations import *
from tqdm import tqdm
import matplotlib.pyplot as plt


train_path = "./data/x_train.csv"
with open(train_path, "r") as f:
    header = np.array(next(csv.reader(f)))

x_train = np.genfromtxt(train_path, delimiter=",", dtype=float, skip_header=1)
y_true = np.genfromtxt("./data/y_train.csv", delimiter=",", dtype=float, skip_header=1)[:,1]
x_test = np.genfromtxt("./data/x_test.csv", delimiter=",", dtype=float, skip_header=1)

nPeople, nFeatures = x_train.shape

def loss(y: np.array, y_true: np.array) -> float:
    return np.mean((y-y_true)**2)

threshold = .75
nPeople, nFeatures = x_train.shape

non_NaN_dominant_cols = np.sum(~np.isnan(x_train), axis=0)
# print(NaN_dominant_cols)
mask_col = non_NaN_dominant_cols/nPeople > threshold
header = header[mask_col]

x_train_filtered_columns = x_train[:, mask_col]
x_test_filtered = x_test[:, mask_col]
print(f'{100*x_train_filtered_columns.shape[1]/nFeatures:.2f}% of columns kept')


non_NaN_dominant_rows = np.sum(~np.isnan(x_train_filtered_columns), axis=1)
mask_row = non_NaN_dominant_rows/x_train_filtered_columns.shape[1] > threshold

x_train_filtered = x_train_filtered_columns[mask_row, :]
y_true = y_true[mask_row]

print(f'{100*x_train_filtered.shape[0]/nPeople:.2f}% of people kept')
print(f"Final shape: {x_train_filtered.shape} (original shape: {x_train.shape})")

x_mean = np.mean(x_train_filtered, axis=0)
x_std = np.std(x_train_filtered, axis=0) + 1e-8
sx = (x_train_filtered - x_mean) / x_std
sx = np.nan_to_num(sx, nan=0.0)

y_mean = np.mean(y_true)
sy = y_true - y_mean

print(sx.shape, sy.shape, (sx.T@sy).squeeze().shape)

nPeople, nFeatures = sx.shape

x_test_mean = np.mean(x_test_filtered, axis=0)
x_test_std = np.std(x_test_filtered, axis=0) + 1e-8
sx_test = (x_test_filtered - x_test_mean) / x_test_std
sx_test = np.nan_to_num(sx_test, nan=0.0)

correlations = np.abs(sx.T@sy)/len(sy)
top_k_mask = correlations > np.percentile(correlations, 70)
sx_filtered = sx[:, top_k_mask]

U, S, Vt = np.linalg.svd(sx_filtered, full_matrices=False)
scores_train = U*S
x_test_scaled = (x_test_filtered - x_mean)/x_std
x_test_scaled = x_test_scaled[:, top_k_mask]
print(U.shape, S.shape, Vt.shape, x_test_scaled.shape)

scores_test = x_test_scaled@Vt.T

k = 15
scores_train_k = scores_train[:, :k]
scores_test_k = scores_test[:, :k]

N_train = scores_train_k.shape[0]
N_test = scores_test_k.shape[0]
train_sq = np.sum(scores_train_k**2, axis=1)
nearest_train_idx = np.zeros(N_test, dtype=int)

test = np.sum(scores_test_k[0:1000], axis=1)


batch_size = 1000

for start_idx in tqdm(range(0, N_test, batch_size), desc="Streaming distances..."):
    end_idx = min(start_idx + batch_size, N_test) #otherwise crashes on last batch

    test_batch = scores_test_k[start_idx:end_idx]
    test_sq_batch = np.sum(test_batch ** 2, axis=1)

    distance = (test_sq_batch[:, None] + train_sq[None, :] - 2 * (test_batch @ scores_train_k.T))
    

    nearest_train_idx[start_idx:end_idx] = np.argmin(distance, axis=1)

y_pred = y_true[nearest_train_idx]
plt.hist(y_pred, alpha=0.3, label="Test")
plt.hist(y_true, alpha=0.3, label="Train")
plt.legend()
plt.show()