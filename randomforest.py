import pickle
import sklearn
import sklearn.ensemble
import pandas as pd
import numpy as np

# how many files to test on
TEST_SIZE = 20
# how many files to train on
TRAIN_SIZE = 200
# how many files to train on at a time
BATCH_SIZE = 5
# how many trees to add per batch
TREES_PER_BATCH = 128

shuffled_files = np.random.permutation(np.arange(1, 675))
train_files = shuffled_files[:TRAIN_SIZE]
test_files = shuffled_files[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

# the features you want to use for training
features = [
    # "vid",
    # "ps",
    # "wkday",
    # "start_time",
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
    "bor0_0",
    "bor0_1",
    "bor0_2",
    "bor0_3",
    "bor0_4",
    "bor0_5",
    "bor1_0",
    "bor1_1",
    "bor1_2",
    "bor1_3",
    "bor1_4",
    "bor1_5",
    "morn",
    "aftnoon",
    "night"
]
# for i in range(24):
#     features.append("hour_{}".format(i))

for i in range(7):
    features.append("wkday_{}".format(i))


test_dfs = []
print("Loading Test Data")
for idx in test_files:
    df = pickle.load(open("PreProcessedData/df_{}.pkl".format(idx), "rb"))
    print("Loaded df_{}".format(idx))
    test_dfs.append(df)
test_df = pd.concat(test_dfs)
test_X = test_df[features]
test_y = test_df["travel_time"]
print(np.mean(test_y))

for feat in features:
    corr = np.corrcoef(test_X[feat], test_y)[0,1]
    print("{}: {}".format(feat, corr))


print("training")
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=0, min_samples_split=2, min_samples_leaf=8,
                                            max_features="sqrt", warm_start=True, n_jobs=-1, verbose=2)
# rf = sklearn.ensemble.RandomForestRegressor()
for idx in range(TRAIN_SIZE // BATCH_SIZE):
    slice = train_files[BATCH_SIZE * idx:BATCH_SIZE * (idx + 1)]
    train_dfs = []
    for i in slice:
        train_dfs.append(pickle.load(open("PreProcessedData/df_{}.pkl".format(i), "rb")))
    df = pd.concat(train_dfs)
    print("Loaded Batch".format(idx))
    X = df[features]
    y = df["travel_time"]

    rf.n_estimators += TREES_PER_BATCH
    rf.fit(X, y)

    if idx % 1 == 0:
        preds = rf.predict(X)
        mean_error = np.mean(np.abs(preds - y))
        print("Training")
        print("Mean Error: {}".format(mean_error))
        rsme = np.sqrt(np.mean(np.square(preds - y)))
        print("RSME: {}".format(rsme))

        preds = rf.predict(test_X)
        mean_error = np.mean(np.abs(preds - test_y))
        print("Testing")
        print("Mean Error: {}".format(mean_error))
        rsme = np.sqrt(np.mean(np.square(preds - test_y)))
        print("RSME: {}".format(rsme))

        sample_mean = np.mean(y)
        mean_error = np.mean(np.abs(sample_mean - test_y))
        print("Naive Testing")
        print("Mean Error: {}".format(mean_error))
        rsme = np.sqrt(np.mean(np.square(sample_mean - test_y)))
        print("RSME: {}".format(rsme))

pickle.dump(rf, open("Models/RandomForest.pkl", "wb"))

