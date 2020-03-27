import linear_regression as lr
import data_loader as dl
import torch

print("n")

t1 = torch.tensor([[1,2,3,4,5,6]])
t2 = torch.tensor([[2,4,6,6,8,3]])

t3 = torch.tensor([[50000.0, 75000.0, 85000.0]])
t4 = torch.tensor([[55000, 85000, 82000]])


assert lr.mse_loss(t1,t2) == torch.tensor([6])
assert lr.mse_loss(t3,t4) == torch.tensor([134000000/3])
assert lr.mae_loss(t3,t4) == torch.tensor([6000])
print("h")

p = torch.empty((1,3))
transpose = torch.tensor([[1.0],[2.8],[3.8]])
print(torch.t(transpose))
print(torch.mm(p,torch.t(t3)))
torch.t(t3)


print(dl.SimpleDataset('C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS1.csv'))
print(dl.get_data_loaders('C:\\Users\\evely\\IntSys-Education\\a2\\data\\DS1.csv'))

