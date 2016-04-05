from group import element
from group import group
from numpy import linalg as LA
import numpy as np
print("dixon starts now")

'''
# C3v
g = [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0]],
     [[0,-1,0,0],[-1,0,0,0],[0,0,1,0]],
    [[-1,1,0,0],[0,1,0,0],[0,0,1,0]], [[1,0,0,0],[1,-1,0,0],[0,0,1,0]],
    [[-1,1,0,0],[-1,0,0,0],[0,0,1,0]],[[0, -1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0]],
]
'''
#'''
#C4v
g = [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0]], [[-1,0,0,0],[0,-1,0,0],[0,0,1,0]],
    [[0,-1,0,0],[1,0,0,0],[0,0,1,0]], [[0,1,0,0],[-1,0,0,0],[0,0,1,0]],
    [[1,0,0,0],[0,-1,0,0],[0,0,1,0]], [[-1,0,0,0],[0,1,0,0],[0,0,1,0]],
    [[0,-1,0,0],[-1,0,0,0],[0,0,1,0]], [[0,1,0,0],[1,0,0,0],[0,0,1,0]]
]
#'''
gk = []

for i in g:
    tmp = element()
    tmp.init(i)
    gk.append(tmp)
G = group()
G.init(gk)
print("multiply table: ", G.mtable)
cl = G.find_class()
print("class: ",cl)
elist = G.subset_product(cl[1],cl[2])
print("element list: ",elist)

H = G.class_mul_constants()
#print("class multiply constants",H)
'''
character_table = G.burnside_class_table()
np.set_printoptions(precision=3)
print("character table: ",character_table)

reg_rep = np.zeros((G.order, G.order, G.order), dtype='int')
for i in range(G.order):
    reg_rep[i,:,:] = G.regular_rep(i)
#print("reg_rep", reg_rep)

order = G.element_order(2)
print("element order: ",order )

eigencolumns = G.reg_eigencolumns(reg_rep[2])
print("eigencolumns: ", eigencolumns)

projection_operator = G.projection_operator(2, character_table, reg_rep)
print("projection_operator", projection_operator*3)

x=np.dot(projection_operator, eigencolumns)
print("after acting on projection operator \n", x)

y = np.dot(reg_rep[3], x)
print("after acting on new reg_rep \n",y)

vec = G.subspace_eigenvector(2, character_table, reg_rep)
print("subspace vector: ", vec)

vec = np.array(vec, dtype='complex').T
print("representation: ", np.dot(np.conj(vec.T), np.dot(reg_rep[1], vec)))
'''

irrep = G.irrep(4)
print("irrep: \n", irrep)