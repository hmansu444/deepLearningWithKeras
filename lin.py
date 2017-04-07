import xlrd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras import optimizers as op
from matplotlib import pyplot as plt


#this class will be used for getting the log
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def createModel(Input, labels):
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=5, init='normal', activation='linear'))# normal for guassian distribution
    model.add(Dense(1, init='normal'))
    # compile model
    adam=op.Adam(lr=0.0009)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    # for weights= layer=Dense(1)
    history = LossHistory()
    model.fit(Input, labels, nb_epoch=4000, batch_size=100, verbose=2 ,callbacks=[history])
    losslist=[]
    for i in range(len(history.losses)) :
        losslist.append(history.losses[i].tolist())
    return model,losslist


workbook = xlrd.open_workbook('data/problem-statement-data/data_carsmall.xlsx')
sheetName = workbook.sheet_by_index(0)

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []

y = []
data = [x1, x2, x3, x4, x5, y]
index = []
nRows = sheetName.nrows
nCols = sheetName.ncols

for row in range(2, nRows):
    value = sheetName.cell_value(row, 5)
    if (value != "NaN"):
        index.append(row)

for row in index:
    i = 0
    for da in data:
        val = sheetName.cell_value(row, i)
        da.append(val)
        i += 1
#print data
for i in range(0, 5):
    data[i] = preprocessing.scale(data[i], with_mean=True, with_std=True)



Input = data[0:5]

labels = y

Input = np.array(Input).transpose()
labels = np.array(y).transpose()

model,lossdata = createModel(Input, labels)

# scores=model.evaluate(Input,labels)

testData = [list(), list(), list(), list(), list()]

for row in range(2, nRows):
    if index.__contains__(row) == False:
        i = 0
        for d in testData:
            val = sheetName.cell_value(row, i)
            d.append(val)
            i += 1
# print testData
for i in range(0, 5):
    testData[i] = preprocessing.scale(testData[i], with_mean=True, with_std=True)

# c=[9.0, 8.0, 454.0, 220.0, 4354.0]
# c=preprocessing.scale(testData,with_mean=True,with_std=True)
#print lossdata
check = np.array(testData).transpose()
y=range(1,len(lossdata)+1)
plt.plot(y,lossdata,'b^')
plt.xlabel('No. of Iterations -->')
plt.ylabel('loss -->')
plt.title('Iterations v/s Loss graph')
#plt.show()
predictions = model.predict(check)
print "Predicted values are :"
print predictions
plt.show()
print "theta values are :"
print model.get_weights()[2],model.get_weights()[3]


