import numpy as np
import csv
from implementations import *

# Import data
train_path = "./data/x_train.csv"
with open(train_path, "r") as f:
    header = np.array(next(csv.reader(f)))

x_train = np.genfromtxt(train_path, delimiter=",", dtype=float, skip_header=1)
y_true = np.genfromtxt("./data/y_train.csv", delimiter=",", dtype=float, skip_header=1)[:,1]
x_test = np.genfromtxt("./data/x_test.csv", delimiter=",", dtype=float, skip_header=1)

# Mean square error loss function
def loss(y: np.array, y_true: np.array) -> float:
    return np.mean((y-y_true)**2)


threshold = .75 # Threshold at which we discard the column/row if there are too many NaN values
nPeople, nFeatures = x_train.shape

non_NaN_dominant_cols = np.sum(~np.isnan(x_train), axis=0) # Count number of NaN values in each column
# print(NaN_dominant_cols)
mask_col = non_NaN_dominant_cols/nPeople > threshold # Discard the columns in which there are too many NaNs
header = header[mask_col]

x_train_filtered_columns = x_train[:, mask_col] 
x_test_filtered = x_test[:, mask_col]
print(f'{100*x_train_filtered_columns.shape[1]/nFeatures:.2f}% of columns kept')


# Filter rows using the same method
non_NaN_dominant_rows = np.sum(~np.isnan(x_train_filtered_columns), axis=1) 
mask_row = non_NaN_dominant_rows/x_train_filtered_columns.shape[1] > threshold

x_train_filtered = x_train_filtered_columns[mask_row, :]
y_true = y_true[mask_row]

print(f'{100*x_train_filtered.shape[0]/nPeople:.2f}% of people kept')
print(f"Final shape: {x_train_filtered.shape} (original shape: {x_train.shape})")

# Reduce and center the distribution s.t. x ~ X(0,1), we could attempt to not center it. I know sometimes it works better but I haven't tried yet.
x_mean = np.mean(x_train_filtered, axis=0)
x_std = np.std(x_train_filtered, axis=0) + 1e-8
sx = (x_train_filtered - x_mean) / x_std

# Change NaN values to 0 otherwise it crashes on multiplication, 
# I think setting them to 0 does the trick but it could be interesting to investigate whether or not it's a good choice
sx = np.nan_to_num(sx, nan=0.0) 

y_mean = np.mean(y_true)
sy = y_true - y_mean

print(sx.shape, sy.shape, (sx.T@sy).squeeze().shape)

nPeople, nFeatures = sx.shape
#x_test_mean = np.mean(x_test_filtered, axis=0)
#x_test_std = np.std(x_test_filtered, axis=0) + 1e-8
#sx_test = (x_test_filtered - x_test_mean) / x_test_std
#sx_test = np.nan_to_num(sx_test, nan=0.0)

# Calculate the correlation factors (first estimate)
correlations = np.abs(sx.T@sy)/len(sy)
top_k_mask = correlations > np.percentile(correlations, 70) # only keep the 70th percentile of the correlations and discard the rest, TODO: try to vary this quantity
sx_filtered = sx[:, top_k_mask]

# Caluclate the singular value decomposition matrices
U, S, Vt = np.linalg.svd(sx_filtered, full_matrices=False)
scores_train = U*S

# Reduce and center the test set with the training mean and std deviation to bring in onto the same system
x_test_scaled = (x_test_filtered - x_mean)/x_std
x_test_scaled = x_test_scaled[:, top_k_mask]
print(U.shape, S.shape, Vt.shape, x_test_scaled.shape)

# Scores are equal to the change of basis matrix multiplied by the test set to bring the test set into the PCA space
scores_test = x_test_scaled@Vt.T

# Pick the top k scores, TODO: see how changing this value changes the results
k = 15
scores_train_k = scores_train[:, :k]
scores_test_k = scores_test[:, :k]

N_train = scores_train_k.shape[0]
N_test = scores_test_k.shape[0]
train_sq = np.sum(scores_train_k**2, axis=1) # Going to be used to calulate the ||.||_2 norm in the batches
nearest_train_idx = np.zeros(N_test, dtype=int) # Stores the index of the component closest to each row of the test set

test = np.sum(scores_test_k[0:1000], axis=1)

# Since the whole thing can't all be in memory, I just batch it and iterate over them. Adjust the batch size according to your RAM and CPU capacities
batch_size = 1000

for start_idx in range(0, N_test, batch_size):
    end_idx = min(start_idx + batch_size, N_test) #otherwise crashes on last batch

    test_batch = scores_test_k[start_idx:end_idx]
    test_sq_batch = np.sum(test_batch ** 2, axis=1)

    # test_sq_batch :(N,), train_sq: (M,) => (N,) * (,M) -> (N,M)
    distance = (test_sq_batch[:, None] + train_sq[None, :] - 2 * (test_batch @ scores_train_k.T)) # Just using that ||v \cdot w ||^2= ||v||^2 + ||w||^2 - 2*<v|w> 
    
    # Get the nearest component from the training set
    nearest_train_idx[start_idx:end_idx] = np.argmin(distance, axis=1)
# Retrieve the corresponding y value of the closes x_train row
y_pred = y_true[nearest_train_idx]
with open("./data/sample_submission.csv", "w") as f:
    f.write("Id,Prediction\n")
    for i in range(y_pred.size):
        f.write(f"{int(x_test[i,0])},{y_pred[i]}\n")