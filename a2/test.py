import linear_regression as lr
import data_loader as dl
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

print("n")

t1 = torch.tensor([[1,2,3,4,5,6]])
t2 = torch.tensor([[2,4,6,6,8,3]])

t3 = torch.tensor([[50000.0, 75000.0, 85000.0]])
t4 = torch.tensor([[55000, 85000, 82000]])


assert lr.mse_loss(t1,t2) == torch.tensor([6])
assert lr.mse_loss(t3,t4) == torch.tensor([134000000/3])
mael=lr.mae_loss(t3, t4)
print(mael)
assert mael == torch.tensor([6000.])
print("h")

p = torch.empty((1,3))
transpose = torch.tensor([[1.0],[2.8],[3.8]])
print(torch.t(transpose))
print(torch.mm(p,torch.t(t3)))
torch.t(t3)


ds1ForActualTesting = dl.SimpleDataset('C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS1.csv') 
'''Apologies for using the filepath from my PC. It wouldn't run properly with the 
local datapath(or maybe I'm doing something wrong ¯\_(ツ)_/¯).'''
dl.get_data_loaders('C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS1.csv')

ds1ForVis = pd.read_csv('C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS2.csv')

#print(ds1ForVis.columns)

#plots the second column of DS1 against the third column of DS1 (uncomment for pretty scatterplot)
#sns.scatterplot(x='x2', y='x3' , data=ds1ForVis) 
#plt.show()

#✧･ﾟ: *✧･ﾟ:*f a n c y*:･ﾟ✧*:･ﾟ✧ 3d plot
#fig = plt.figure().gca(projection = '3d')
#fig.scatter(ds1ForVis['x1'], ds1ForVis['x2'], ds1ForVis['x3'])
#plt.show()

#actual testing, hopefully
print(ds1ForActualTesting.__getitem__(1))
labels = []
targets = []
for i in range(0, ds1ForActualTesting.data.shape[0]):
    labels.append(ds1ForActualTesting.__getitem__(i)[0])
    targets.append(ds1ForActualTesting.__getitem__(i)[1])
path_to_csv = 'C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS1.csv'
transform_fn = lr.data_transform  # None would've
train_val_test = [0.6, 0.2, 0.2]
batch_size = 32
num_param = 3
learnrt = 0.01
loss_fn = lr.mae_loss
TOTAL_TIME_STEPS = 100

train_loader, val_loader, test_loader =\
        dl.get_data_loaders(
            path_to_csv,
            transform_fn=transform_fn,
            train_val_test=train_val_test,
            batch_size=batch_size)

model = lr.LinearRegressionModel(num_param)
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()

for t in range(TOTAL_TIME_STEPS):
    for batch_index, (input_t, y) in enumerate(train_loader):

        optimizer.zero_grad()
        #print(model.thetas)
        preds = model(input_t)  
        #print(preds)
        #print(y)
        loss = loss_fn(preds, y.view(1,len(y)))  # You might have to change the shape of things here.
        loss.backward()
        #print(loss)
        optimizer.step()

model.eval()
for batch_index, (input_t, y) in enumerate(val_loader):

    preds = model(input_t)

    loss = loss_fn(preds, y.view(1,len(y)))
    print(loss)