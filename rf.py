import pickle
import sklearn
import sklearn.ensemble
import pandas as pd
import numpy as np

# how many files to test on
TEST_SIZE = 1
# how many files to train on
TRAIN_SIZE = 670
# how many files to train on at a time
BATCH_SIZE = 10
# how many trees to add per batch
TREES_PER_BATCH = 8

shuffled_files = np.arange(1, 675)#np.random.permutation(np.arange(1, 675))
train_files = shuffled_files[TEST_SIZE:TRAIN_SIZE+TEST_SIZE]
test_files = shuffled_files[ 0:TEST_SIZE]

# the features you want to use for training
features = [
    # "vid",
    # "ps",
    "start_time",
    "euc_dist",
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
    # "bor0_0",
    # "bor0_1",
    # "bor0_2",
    # "bor0_3",
    # "bor0_4",
    # "bor0_5",
    # "bor1_0",
    # "bor1_1",
    # "bor1_2",
    # "bor1_3",
    # "bor1_4",
    # "bor1_5",
    "wkd",
    # "h",
]
# for i in range(24):
#     features.append("hour_{}".format(i))
#
# for i in range(7):
#     features.append("wkday_{}".format(i))

for i in range(6):
    for j in range(6):
        features.append("bor{}to{}".format(i,j))

# for i in range(2,4+1):
#     features.append('wkday^{}'.format(i))
# for i in range(2,6+1):
#     features.append('hour^{}'.format(i))
# for i in range(14):
#     features.append('weatherType_{}'.format(i))


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




def reportErr(err,attention=''):
    l2e = np.linalg.norm(err)/np.sqrt(len(err))
    l1e = np.mean(np.abs(err))
    median_e = np.median(np.abs(err))
    print("RMSE(L2): {}".format( l2e ))
    print(attention+"L1 Error: {}".format( l1e ))
    print("Median Error: {}".format( median_e ))
    return l2e,l1e,median_e






N_samples, L2E, MeanE, MedianE  = [], [], [], []
_stp = int(100000/20)
for num in [100000]:# range(_stp,100000,_stp):#
    print("\n\n\ntraining-----------------",num)
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=0, min_samples_split=4, min_samples_leaf=8,
                                                max_features="auto", warm_start=True, n_jobs=-1, verbose=0)
    for idx in range(TRAIN_SIZE // BATCH_SIZE):
        slice = train_files[BATCH_SIZE * idx:BATCH_SIZE * (idx + 1)]
        train_dfs = []
        for i in slice:
            train_dfs.append(pickle.load(open("PreProcessedData_eagle/df_{}.pkl".format(i), "rb")))
        df = pd.concat(train_dfs)
        df = df[df['travel_time'] >= 60]
        df = df[df['travel_time'] <= 3600 * 4]
        print("Loaded Batch - {}".format(idx))
        #df = df.iloc[:num, :]
        X = df[features]
        y = df["travel_time"]

        rf.n_estimators += TREES_PER_BATCH
        rf.fit(X, y)

        if idx % 1 == 0:
            print("Training Error")
            reportErr(rf.predict(X)-y)

            #preds = [MLP.predict(test_X) for MLP in MLP_list]
            #preds = np.mean(preds, axis=0)
            print("Testing Error")
            l2e,l1e,me=reportErr(rf.predict(test_X)-test_y,'****************')
            L2E.append(l2e)
            MeanE.append(l1e)
            MedianE.append(me)
            N_samples.append(num)

            print("Naive Testing:")
            reportErr(np.mean(y)-test_y)

#np.save('./rf_study_2',[N_samples,L2E,MeanE,MedianE])

#pickle.dump(rf, open("Models/RandomForest.pkl", "wb"))
