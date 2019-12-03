import pickle
import sklearn
import sklearn.ensemble
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
    "wkd",
    'h',
    #"start_time",
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
]
for i in range(2,2+1):
    features.append('wkday^{}'.format(i))
for i in range(2,4+1):
    features.append('hour^{}'.format(i))
for i in range(6):
    for j in range(6):
        features.append("bor{}to{}".format(i,j))
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
_stp = 2
for num in [5000]:#range(_stp,120,_stp):#
    print("training-----------------",num)
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
        df = df.iloc[:num, :]
        X = df[features]
        y = df["travel_time"]

        lr.fit(X, y)


        if idx % 1 == 0:
            print("Training Error")
            reportErr(lr.predict(X)-y)

            #preds = [MLP.predict(test_X) for MLP in MLP_list]
            #preds = np.mean(preds, axis=0)
            print("Testing Error")
            l2e,l1e,me=reportErr(lr.predict(test_X)-test_y,'**********************************')
            L2E.append(l2e)
            MeanE.append(l1e)
            MedianE.append(me)
            N_samples.append(num)

            print("Naive Testing:")
            reportErr(np.mean(y)-test_y)

#np.save('./lr_study',[N_samples,L2E,MeanE,MedianE])
#pickle.dump(lr, open("Models/RandomForest.pkl", "wb"))
