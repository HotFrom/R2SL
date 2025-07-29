import numpy as np
import pandas as pd
import math
import datetime

# ---------------- Configuration ----------------
TRAIN_PATH = '0.02_rt_lda5_train.txt'
OUTPUT_TRAIN = '0.02_rt_m2lda5_train.txt'
NUM_HIDDEN_STATES = 5
PENALTY_INIT = 50
MAX_ITER = 15
SEED = 10

# ---------------- Load training data ----------------
train = pd.read_csv(TRAIN_PATH, sep='\t', header=None)
UserCountry = np.array(train.iloc[:, 3])
ServiceCountry = np.array(train.iloc[:, 5])
UserAS = np.array(train.iloc[:, 4])
ServiceAS = np.array(train.iloc[:, 6])
QoS = np.round(np.array(train.iloc[:, 2]), 3)

# Normalize QoS to [0, 10000] scale
QoS = ((QoS - QoS.min()) / (QoS.max() - QoS.min())) * 20 * 500

# Record format: [UserCountry, ServiceCountry, QoS, UserAS, ServiceAS]
record = np.zeros((len(UserCountry), 5))
for i in range(len(UserCountry)):
    record[i][0] = int(UserCountry[i])
    record[i][1] = int(ServiceCountry[i])
    record[i][2] = QoS[i]
    record[i][3] = int(UserAS[i])
    record[i][4] = int(ServiceAS[i])

# ---------------- Initialization ----------------
N = len(record)
M = NUM_HIDDEN_STATES
w = PENALTY_INIT
np.random.seed(SEED)

The_a = np.ones((M, 1)) / M
The_e = np.ones((M, 1)) / M

T_ijk = np.zeros((N, M, M))
t_ijk = np.zeros((N, M, M))
The_ijk = np.zeros((N, M, M))

Ba = np.random.rand(M, int(max(UserCountry)) + 100)
Be = np.random.rand(M, int(max(ServiceCountry)) + 100)
Ba_2 = np.random.rand(M, int(max(UserAS)) + 100)
Be_2 = np.random.rand(M, int(max(ServiceAS)) + 100)

Ca = np.ones((int(max(UserCountry)) + 10, 1)) * 10
Ce = np.ones((int(max(ServiceCountry)) + 10, 1)) * 10
Ca_2 = np.ones((int(max(UserAS)) + 10, 1)) * 10
Ce_2 = np.ones((int(max(ServiceAS)) + 10, 1)) * 10

# ---------------- Core functions ----------------
def compute_t_ijk():
    for i in range(N):
        for j in range(M):
            for k in range(M):
                t_ijk[i, j, k] = (
                    Ba[j, int(record[i, 0])] *
                    Be[k, int(record[i, 1])] *
                    Ba_2[j, int(record[i, 3])] *
                    Be_2[k, int(record[i, 4])]
                )

def compute_The_ijk():
    for i in range(N):
        for j in range(M):
            for k in range(M):
                if record[i, 2] < 250:
                    nu = Ca[int(record[i, 0])] * Ce[int(record[i, 1])] * Ca_2[int(record[i, 3])] * Ce_2[int(record[i, 4])]
                else:
                    nu = Ca[int(record[i, 0])] * Ce[int(record[i, 1])] * w * Ca_2[int(record[i, 3])] * Ce_2[int(record[i, 4])]
                result = (1 / nu) * np.exp(-record[i, 2] / nu)
                The_ijk[i, j, k] = result

def expectation_step():
    for i in range(N):
        for j in range(M):
            for k in range(M):
                T_ijk[i, j, k] = t_ijk[i, j, k] * The_ijk[i, j, k]
        T_ijk_sum = T_ijk[i].sum()
        T_ijk[i] = T_ijk[i] / T_ijk_sum if T_ijk_sum > 0 else T_ijk[i]
    return T_ijk

def maximization_step():
    for i in range(M):
        The_a[i] = ((4 * N) + T_ijk[:, i, :].sum()) / ((4 * M + 1) * N)
        The_e[i] = ((4 * N) + T_ijk[:, :, i].sum()) / ((4 * M + 1) * N)

    for i in range(M):
        for q in range(Ba.shape[1]):
            Ba[i, q] = T_ijk[np.where(record[:, 0] == q)[0], i, :].sum()
        for q in range(Be.shape[1]):
            Be[i, q] = T_ijk[np.where(record[:, 1] == q)[0], :, i].sum()
        for q in range(Ba_2.shape[1]):
            Ba_2[i, q] = T_ijk[np.where(record[:, 3] == q)[0], i, :].sum()
        for q in range(Be_2.shape[1]):
            Be_2[i, q] = T_ijk[np.where(record[:, 4] == q)[0], :, i].sum()

        Ba[i] = np.round(Ba[i] / (Ba[i].sum() + 1e-8) * Ba.shape[1], 4)
        Be[i] = np.round(Be[i] / (Be[i].sum() + 1e-8) * Be.shape[1], 4)
        Ba_2[i] = np.round(Ba_2[i] / (Ba_2[i].sum() + 1e-8) * Ba_2.shape[1], 4)
        Be_2[i] = np.round(Be_2[i] / (Be_2[i].sum() + 1e-8) * Be_2.shape[1], 4)

def gradient_descent():
    global w
    epsilon = 1e-10
    for i in range(N):
        for j in range(M):
            for k in range(M):
                ca = Ca[int(record[i, 0])]
                ce = Ce[int(record[i, 1])]
                if j == k:
                    denom = ce * ca ** 2
                else:
                    denom = ce * ca ** 2 * w
                denom = max(denom, epsilon)
                delta = T_ijk[i, j, k] * (record[i, 2] / denom - 1 / ca)
                Ca[int(record[i, 0])] += delta * 0.01
    for i in range(N):
        for j in range(M):
            for k in range(M):
                ca = Ca[int(record[i, 0])]
                ce = Ce[int(record[i, 1])]
                if j == k:
                    denom = ca * ce ** 2
                else:
                    denom = ca * ce ** 2 * w
                denom = max(denom, epsilon)
                delta = T_ijk[i, j, k] * (record[i, 2] / denom - 1 / ce)
                Ce[int(record[i, 1])] += delta * 0.01

    total_gradient = 0
    for i in range(N):
        for j in range(M):
            for k in range(M):
                if j != k:
                    denom = Ca[int(record[i, 0])] * Ce[int(record[i, 1])] * w
                    denom = max(denom, epsilon)
                    total_gradient += T_ijk[i, j, k] * (-1 / w + record[i, 2] / denom)
    w += 0.001 * total_gradient

def loss():
    return t_ijk.sum() * The_ijk.sum() * 10

# ---------------- Training Loop ----------------
loss_list = []
l_prev = 1
l_curr = 25
iteration = 0

while iteration < MAX_ITER:
    start_time = datetime.datetime.now()
    iteration += 1
    compute_t_ijk()
    compute_The_ijk()
    expectation_step()
    maximization_step()
    gradient_descent()
    loss_delta = abs(l_curr - l_prev) / l_curr
    loss_list.append(loss_delta)
    print(f"Iteration {iteration} | Loss delta: {loss_delta:.6f}")
    l_prev, l_curr = l_curr, loss()
    print(f"Time taken: {(datetime.datetime.now() - start_time).total_seconds():.2f}s")

# ---------------- Save output: train ----------------
with open(OUTPUT_TRAIN, 'w') as f:
    for i in range(len(record)):
        x = int(record[i][0])
        x2 = int(record[i][3])
        u = int(record[i][1])
        u2 = int(record[i][4])
        f.write(f"{int(i)}\t{i}\t{record[i][2]:.3f}\t")
        f.write(f"{x}\t{x2}\t{u}\t{u2}\t")
        f.write('\t'.join(map(str, Ba[:, x])) + '\t')
        f.write('\t'.join(map(str, Ba_2[:, x2])) + '\t')
        f.write('\t'.join(map(str, Be[:, u])) + '\t')
        f.write('\t'.join(map(str, Be_2[:, u2])) + '\n')

# ---------------- Save output: test ----------------
test = pd.read_csv(TEST_PATH, sep='\t', header=None)
userID2 = np.array(test.iloc[:, 0])
serviceID2 = np.array(test.iloc[:, 1])
QoS2 = np.round(np.array(test.iloc[:, 2]), 3)
UserCountry2 = np.array(test.iloc[:, 3])
UserAS2 = np.array(test.iloc[:, 4])
ServiceCountry2 = np.array(test.iloc[:, 5])
ServiceAS2 = np.array(test.iloc[:, 6])

with open(OUTPUT_TEST, 'w') as f:
    for i in range(len(userID2)):
        x = int(UserCountry2[i])
        x2 = int(UserAS2[i])
        u = int(ServiceCountry2[i])
        u2 = int(ServiceAS2[i])
        f.write(f"{int(userID2[i])}\t{int(serviceID2[i])}\t{QoS2[i]:.3f}\t")
        f.write(f"{x}\t{x2}\t{u}\t{u2}\t")
        f.write('\t'.join(map(str, Ba[:, x])) + '\t')
        f.write('\t'.join(map(str, Ba_2[:, x2])) + '\t')
        f.write('\t'.join(map(str, Be[:, u])) + '\t')
        f.write('\t'.join(map(str, Be_2[:, u2])) + '\n')

print("All done. Outputs written to file.")
