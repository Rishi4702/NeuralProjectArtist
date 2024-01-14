import torch
import numpy as np

# tensors
# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)
#
# print(x_data)
# np_array = np.array(data)
# print(np_array.shape,np_array)
#
# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")
#
# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

# shape = (3,3,3)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
#
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# tensor = torch.rand(9,9)
# # We move our tensor to the GPU if available
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print("hello")
#
# # tensor = torch.ones(4, 4)
# # # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# # y1 = tensor @ tensor.T
# # y2 = tensor.matmul(tensor.T)
# #
# # y3 = torch.rand_like(tensor)
# # torch.matmul(tensor, tensor.T, out=y3)
# #
# #
# # # This computes the element-wise product. z1, z2, z3 will have the same value
# # z1 = tensor * tensor
# # z2 = tensor.mul(tensor)
# #
# # z3 = torch.rand_like(tensor)
# # torch.mul(tensor, tensor, out=z3)
# #
# # print(y3,"\n",z3)
# tensor = torch.ones(4, 4)
# agg_item = tensor.sum().item()
#
# print(agg_item)
# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
#
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")
# dataSETS and dataloaders
# x = torch.ones(5)  # input tensor
# y = torch.zeros(3)  # expected output
# print(x.shape,x)
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
# print(w.shape,w)
# z = torch.matmul(x, w)+b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

