import pickle
import sklearn
import sklearn.ensemble
import sklearn.neural_network as nn
import pandas as pd
import numpy as np

# how many files to test on
TEST_SIZE = 1
# how many files to train on
TRAIN_SIZE = 1
# how many files to train on at a time
BATCH_SIZE = 1

#shuffled_files = np.random.permutation(np.arange(1, 675))
train_files = 1+np.arange(TEST_SIZE,TEST_SIZE+TRAIN_SIZE)#shuffled_files[:TRAIN_SIZE]
test_files = 1+np.arange(TEST_SIZE)#shuffled_files[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

# the features you want to use for training
features = [
    # "vid",
    # "ps",
    # "wkday",
    "start_time",
    # "euc_dist",
    "real_dist",
    # "humidity",
    # "windspeed",
    # "vis",
    # "temp",
    # "haze",
    # "fog",
    # "rain",
    # "snow",
    # "hday",
]
# for i in range(2,4+1):
#     features.append('wkday^{}'.format(i))
# for i in range(2,6+1):
#     features.append('hour^{}'.format(i))
for i in range(6):
    for j in range(6):
        features.append("bor{}to{}".format(i,j))
# for i in range(14):
#     features.append('weatherType_{}'.format(i))


def reportErr(err,attention=''):
    print("RMSE(L2): {}".format( np.linalg.norm(err)/np.sqrt(len(err)) ))
    print(attention+"L1 Error: {}".format( np.mean(np.abs(err)) ))
    print("Median Error: {}".format( np.median(np.abs(err)) ))



test_dfs = []
print("Loading Test Data")
for idx in test_files:
    df = pickle.load(open("PreProcessedData_eagle/df_{}.pkl".format(idx), "rb"))
    print("Loaded df_{}".format(idx))
    test_dfs.append(df)
test_df = pd.concat(test_dfs)
test_df = test_df[test_df['travel_time'] >= 60]
test_df = test_df[test_df['travel_time'] <= 3600*4]
test_X = test_df[features]
test_y = test_df["travel_time"]
print('travel_time - max, min, mean:',np.max(test_y),np.min(test_y),np.mean(test_y))




for feat in features:
    corr = np.corrcoef(test_X[feat], test_y)[0,1]
    print("{}: {}".format(feat, corr))


print("\n\n\n\n\n\n\ntraining")

MLP_list = []
mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(10,3),verbose=True,max_iter=20,learning_rate_init=0.1,warm_start=True,early_stopping=True,tol=0.00010)

for idx in range(TRAIN_SIZE // BATCH_SIZE):
    slice = train_files[BATCH_SIZE * idx:BATCH_SIZE * (idx + 1)]
    train_dfs = []
    for i in slice:
        train_dfs.append(pickle.load(open("PreProcessedData_eagle/df_{}.pkl".format(i), "rb")))
    df = pd.concat(train_dfs)
    df = df[df['travel_time'] >= 60]
    df = df[df['travel_time'] <= 3600 * 0.5]
    print("Loaded Batch-{}".format(idx))
    X = df[features]
    y = df["travel_time"]

    mlp.fit(X, y)
    #MLP_list.append(mlp)

    if idx % 1 == 0:
        print("Training Error")
        reportErr(mlp.predict(X)-y)

        #preds = [MLP.predict(test_X) for MLP in MLP_list]
        #preds = np.mean(preds, axis=0)
        print("Testing Error")
        reportErr(mlp.predict(test_X)-test_y,'****************')

        print("Naive Testing:")
        reportErr(np.mean(y)-test_y)

#pickle.dump(MLP_list, open("Models/LinearRegression.pkl", "wb"))
