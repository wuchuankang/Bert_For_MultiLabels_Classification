import numpy
import torch as t

a = t.randn(2,3)
b = [a for _ in range(2)]

print(t.tensor([l.numpy() for l in b]))
