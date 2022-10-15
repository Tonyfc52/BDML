
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend
import matplotlib.pyplot as plt


def recalculate(moddata, newmod):

    # ScalerSet
    minvalue_high = moddata['high'].min()
    minvalue_close = moddata['close'].min()
    maxvalue_high = moddata['high'].max()
    maxvalue_close = moddata['close'].max()

    # 設定類似K值天數(n!=0)，含當天
    n = 5
    leng = len(moddata)
    TodayK = 50
    PreviousK = 0
    K, B, R, xH, tP = [], [], [], [], []

    for i in range(n-1, leng):
        # N日的類K值
        highT = moddata.iloc[i-n+1:i+1].values.max()
        lowT = moddata.iloc[i-n+1:i+1].values.min()

        RSV = (moddata['close'].iloc[i]-lowT)/(highT-lowT)*100

        if i > (n-1):
            TodayK = PreviousK*2/3+RSV/3
        else:
            TodayK = 50

        K.append(TodayK)
        PreviousK = TodayK

        # 因為價格越高價差越大，所以改成K bar body部份的占比
        OtoC = abs(moddata['close'].iloc[i]-moddata['open'].iloc[i])
        HtoL = moddata['high'].iloc[i]-moddata['low'].iloc[i]
        if HtoL == 0:
            B.append(0)
        else:
            B.append(OtoC/HtoL)

        # 漲跌幅度(跟前一天的比，並以漲跌幅10%,-10%做normalized)
        preNormal = (moddata['close'].iloc[i] -
                     moddata['close'].iloc[i-1])/moddata['close'].iloc[i-1]
        normalized = (preNormal-(-0.1))/(0.1-(-0.1))
        R.append(normalized)

        # normalized金額(和最低值相比)
        xH.append((moddata['high'].iloc[i]-minvalue_high) /
                  (maxvalue_high-minvalue_high))
        tP.append((moddata['close'].iloc[i]-minvalue_close) /
                  (maxvalue_close-minvalue_close)*0.6+0.2)

    # 之所以留下high發現在decisitionTree時，計算feature importance四個價位以這個權重最高
    newmod['high'] = xH
    newmod['Body'] = B
    newmod['Kvalue'] = K
    newmod['todayP'] = tP
    newmod['Result'] = R
    return (newmod)


def withshare(S, P):
    act, hold = 0, 0
    if S == 1 or S == 0:  # 持有時當日上漲
        if P == 1 or P == 0:
            act = 0
            hold = 1

        if P == -1:
            act = -1
            hold = 0

    if S == -1:  # 持有時當日下跌
        if P == 1:
            act = 0
            hold = 1
        if P == -1 or P == 0:  # 認賠
            act = -1
            hold = 0

    return (act, hold)


def shortshare(S, P):
    act, hold = 0, 0
    if S == 1:  # 做空時當日上漲
        if P == 1:
            act = 1  # 認賠
            hold = 0

        if P == -1 or P == 0:
            act = 0
            hold = -1

    if S == -1 or S == 0:  # 做空時當日下跌
        if P == 1:
            act = 1
            hold = 0
        if P == -1 or P == 0:
            act = 0
            hold = -1
    return (act, hold)


def empty(S, P, Restday):
    act, hold = 0, 0

    if P == 1:
        act = 1
        hold = 1
    if P == -1:
        act = -1
        hold = -1
    return (act, hold)


# You can write code above the if-main block.
if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv",
                        help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv",
                        help="input testing data file name")
    parser.add_argument("--output", default="output.csv",
                        help="output file name")
    args = parser.parse_args()


# The following part is an example.
# You can modify it at will.
training_file = args.training
testing_file = args.testing


########                      Start From Here                 ########
training = pd.read_csv(training_file, header=None)
testing = pd.read_csv(testing_file, header=None)

# 知道test資料的筆數
testcount = len(testing)

# 兩筆連續性資料merge
overall = pd.merge(training, testing, how='outer')
overall.columns = ['open', 'high', 'low', 'close']


# high最高值，Body棒本體比例，Kvalue參考K值，todayP今日收盤價，Result與前一日收盤價的漲跌
trainset = pd.DataFrame(
    [], columns=['high', 'Body', 'Kvalue', 'todayP', 'Result'])
testset = pd.DataFrame(
    [], columns=['high', 'Body', 'Kvalue', 'todayP', 'Result'])
totalset = pd.DataFrame(
    [], columns=['high', 'Body', 'Kvalue', 'todayP', 'Result'])

recalculate(overall, totalset)

# 切割資料
trainset = totalset[0:(-1*testcount-1)]
testset = totalset[(-1*testcount-1):]

#----------Train Data-------#

tempX = trainset.drop(['todayP', 'Result'], axis=1)
tempY = trainset['todayP']
training_length = len(tempX)
trainX, trainY = [], []

# 考慮的天數 (day=1等於只考慮當天)
day = 1
for i in range(day-1, training_length-1):
    trainX.append(tempX.iloc[i-day+1:i+1].values)
    trainY.append(tempY.iloc[i+1])


# Keras Training preparation

trainX = np.array(trainX)
trainY = np.array(trainY)

#---------開始train---------#

features = len(tempX.columns)
stepdays = len(trainX[0])


# 若先前有跑過則清乾淨
backend.clear_session()

model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=False,
          dropout=0.2, input_shape=(stepdays, features)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.summary()
history = model.fit(trainX, trainY, epochs=600, verbose=0, batch_size=20)

# Evaluation of training
test_length = len(testset)
testX = []
testY = []
truelist = []
tempX = testset.drop(['todayP', 'Result'], axis=1)
tempY = testset['todayP']


# 考慮的天數=training天數(1等於當天)
day = 1
for i in range(day-1, test_length-1):
    testX.append(tempX.iloc[i-day+1:i+1].values)
    testY.append(tempY.iloc[i+1])

truelist = testY
testX = np.array(testX)
testY = np.array(testY)

pred = model.predict(testX)

predlist = []
for i in range(len(pred)):
    predlist.append(pred[i][0])

#comp = pd.DataFrame({"prediction": predlist, "Truth": truelist})
# comp.plot()
# plt.show()

# 改寫成正負號
predlist = [np.sign(predlist[j]-predlist[j-1])
            for j in range(1, len(predlist))]
truelist = [np.sign(truelist[j]-truelist[j-1])
            for j in range(1, len(truelist))]

#Prediction and Action
Days = len(truelist)

hold = 0
act, h = 0, 0  # 暫存動作及狀態
action = []
behavior = ['Short/Sell', 'Hold', 'Buy']

# Read Day1
D0Stat = truelist[0]  # 當日股價
D1Pred = predlist[0]  # 隔日預測

if D1Pred == 1 or D1Pred == 0:
    action.append(1)  # 買進
    hold = 1  # 隔天持有股票數

if D1Pred == -1:
    action.append(-1)  # 做空
    hold = -1  # 隔天持有股票數

print(
    f'Day  1 Tomorrow action:{behavior[action[0]+1]:10}, Share will be:{hold}')

# NextDay(s)
for i in range(1, Days):
    D0Stat = truelist[i]  # 當天狀態
    D1Pred = predlist[i]
    if hold == 1:
        act, h = withshare(D0Stat, D1Pred)

    if hold == -1:
        act, h = shortshare(D0Stat, D1Pred)

    if hold == 0:
        act, h = empty(D0Stat, D1Pred, Days-i)

    action.append(act)
    hold = h
    print(
        f'Day {i+1:2} Tomorrow action:{behavior[act+1]:10}, Share will be:{h}')


print('Writing Final Action File....')

with open(args.output, "w") as output_file:
    for a in action:
        output_file.write(str(a)+'\n')
