import torch

x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
'''
#print(mathit(9, 1)) # math it directly handled

outcome = mathit(9,1) #mathit inderictely hadled by var

print(outcome * 20) # function var handled by var can be 
'''					# manipulated
outcome = x*y
print(outcome)

x = torch.zeros([2,5])

print(x)
#tensor is a multidimensional array

x.shape
torch.Size([2, 5])

y = torch.rand([2,5])

print (y)

y = y.view([1,10])

print(y)