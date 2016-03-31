from group import element
from group import group
from numpy import linalg as LA
import numpy as np
print("dixon starts now")

'''
# C3v
g = [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0]], [[0,-1,0,0],[1,-1,0,0],[0,0,1,0]],
    [[-1,1,0,0],[-1,0,0,0],[0,0,1,0]], [[0,-1,0,0],[-1,0,0,0],[0,0,1,0]],
    [[-1,1,0,0],[0,1,0,0],[0,0,1,0]], [[1,0,0,0],[1,-1,0,0],[0,0,1,0]],

]
'''
'''
#C4v
g = [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0]], [[-1,0,0,0],[0,-1,0,0],[0,0,1,0]],
    [[0,-1,0,0],[1,0,0,0],[0,0,1,0]], [[0,1,0,0],[-1,0,0,0],[0,0,1,0]],
    [[1,0,0,0],[0,-1,0,0],[0,0,1,0]], [[-1,0,0,0],[0,1,0,0],[0,0,1,0]],
    [[0,-1,0,0],[-1,0,0,0],[0,0,1,0]], [[0,1,0,0],[1,0,0,0],[0,0,1,0]]
]
'''
gk = []

for i in g:
    tmp = element()
    tmp.init(i)
    gk.append(tmp)
G = group()
G.init(gk)
cl = G.find_class()
print(cl)
elist = G.subset_product(cl[1],cl[2])
print(elist)

H = G.class_mul_constants()
print(H)

character_table = G.burnside_class_table()
print(character_table)

'''
h = H[3] + H[2]
print("some h",h)
w,v = LA.eig(h)
print(w)
print(v)
'''