import xlrd
import numpy as np
from  keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn import preprocessing as p


def baseline_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, init='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
np.random.seed(seed)

x1=[]
x2=[]
y=[]

tx1=[]
tx2=[]

xl_workbook1 = xlrd.open_workbook("data/problem-statement-date/ex2data1-logistic.xls")
xl_workbook2 = xlrd.open_workbook("data/problem-statement-date/ex2data2-logistic.xls")

xl_sheet1 = xl_workbook1.sheet_by_index(0)
xl_sheet2 = xl_workbook2.sheet_by_index(0)


for row_idx in range(1, int(xl_sheet1.nrows*.90)):
    x1.append(xl_sheet1.cell(row_idx, 0).value)
    x2.append(xl_sheet1.cell(row_idx, 1).value)
    y.append(xl_sheet1.cell(row_idx, 2).value)
for row_idx in range(1, int(xl_sheet2.nrows*.90)):
    x1.append(xl_sheet2.cell(row_idx, 0).value)
    x2.append(xl_sheet2.cell(row_idx, 1).value)
    y.append(xl_sheet2.cell(row_idx, 2).value)

for row_idx in range(int(xl_sheet1.nrows*.90),xl_sheet1.nrows):
    tx1.append(xl_sheet1.cell(row_idx, 0).value)
    tx2.append(xl_sheet1.cell(row_idx, 1).value)

for row_idx in range(int(xl_sheet2.nrows*.90),xl_sheet2.nrows):
    tx1.append(xl_sheet2.cell(row_idx, 0).value)
    tx2.append(xl_sheet2.cell(row_idx, 1).value)


x1=p.scale(x1)
x2=p.scale(x2)

features=[x1,x2]

features=np.array(features).transpose()
y=np.array(y).transpose()

testFeatures = [tx1,tx2]
testFeatures = np.array(testFeatures).transpose()

model=baseline_model()
model.fit(features, y, nb_epoch=800, batch_size=100,  verbose=2)
# calculate predictions
predictions = model.predict(testFeatures)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print((rounded))
for i in range(0,len(rounded)):
    print int(rounded[i]),
