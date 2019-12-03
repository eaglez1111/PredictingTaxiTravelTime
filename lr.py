import pickle
import sklearn
import sklearn.ensemble
import pandas as pd
import numpy as np

# how many files to test on
TEST_SIZE = 10
# how many files to train on
TRAIN_SIZE = 100
# how many files to train on at a time
BATCH_SIZE = 1

#shuffled_files = np.random.permutation(np.arange(1, 675))
train_files = 1+np.arange(TEST_SIZE,TEST_SIZE+TRAIN_SIZE)#shuffled_files[:TRAIN_SIZE]
test_files = 1+np.arange(TEST_SIZE)#shuffled_files[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]


# the features you want to use for training
features = [
    "vid",
    "ps",
    "wkday",
    "start_time",
    # "euc_dist",
    "real_dist",
    "humidity",
    "windspeed",
    "vis",
    "temp",
    "haze",
    "fog",
    "rain",
    "snow",
    "hday",
]
for i in range(2,4+1):
    features.append('wkday^{}'.format(i))
for i in range(2,6+1):
    features.append('hour^{}'.format(i))
for i in range(6):
    for j in range(6):
        features.append("bor{}to{}".format(i,j))



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
print(np.sort(test_y)[-1000:-1])
print(np.mean(test_y))
print(np.mean(test_y) + 4*np.std(test_y))
print(np.percentile(test_y, [0.1, 99.9]))



for feat in features:
    corr = np.corrcoef(test_X[feat], test_y)[0,1]
    print("{}: {}".format(feat, corr))


print("training")
lr = sklearn.linear_model.LinearRegression()
for idx in range(TRAIN_SIZE // BATCH_SIZE):
    slice = train_files[BATCH_SIZE * idx:BATCH_SIZE * (idx + 1)]
    train_dfs = []
    for i in slice:
        train_dfs.append(pickle.load(open("PreProcessedData_eagle/df_{}.pkl".format(i), "rb")))
    df = pd.concat(train_dfs)
    df = df[df['travel_time'] >= 60]
    df = df[df['travel_time'] <= 3600 * 0.5]
    print("Loaded Batch".format(idx))
    X = df[features]
    y = df["travel_time"]

    lr.fit(X, y)

    if idx % 1 == 0:
        preds = lr.predict(X)
        mean_error = np.mean(np.abs(preds - y))
        print("Training")
        print("Mean Error: {}".format(mean_error))
        print("Median Error: {}".format(np.median(np.abs(preds - y))))
        rsme = np.sqrt(np.mean(np.square(preds - y)))
        print("RSME: {}".format(rsme))

        preds = lr.predict(test_X)
        mean_error = np.mean(np.abs(preds - test_y))
        print("Testing")
        print("Mean Error: {}".format(mean_error))
        print("Median Error: {}".format(np.median(np.abs(preds - test_y))))
        print("Max Error: {}".format(np.max(np.abs(preds - test_y))))
        print(np.max(test_y))
        print(np.min(test_y))
        rsme = np.sqrt(np.mean(np.square(preds - test_y)))
        print("RSME: {}".format(rsme))

        sample_mean = np.mean(y)
        mean_error = np.mean(np.abs(sample_mean - test_y))
        print("Naive Testing")
        print("Mean Error: {}".format(mean_error))
        rsme = np.sqrt(np.mean(np.square(sample_mean - test_y)))
        print("RSME: {}".format(rsme))

pickle.dump(lr, open("Models/RandomForest.pkl", "wb"))
