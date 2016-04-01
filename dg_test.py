from double_group import element
from double_group import group
from numpy import linalg as LA
import numpy as np
print("double group test starts now")

#D4
g = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]], [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]],
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]], [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0]]

]

s = [
    [[1,0],[0,1]], [[-1j, 0], [0,1j]], [[(1-1j)/np.sqrt(2), 0], [0, (1+1j)/np.sqrt(2)]], [[(1+1j)/np.sqrt(2), 0], [0, (1-1j)/np.sqrt(2)]],
    [[0,-1],[1,0]], [[0,-1j],[-1j,0]], [[0,-(1+1j)/np.sqrt(2)], [(1-1j)/np.sqrt(2), 0]], [[0,(1-1j)/np.sqrt(2)], [-(1+1j)/np.sqrt(2),0]]
]

#print(np.array(s))

s1 = [[[1,0],[0,1]], [[-1j, 0],[0,1j]],[[0,-1],[1,0]], [[0,-1j], [-1j,0]]]

#'''
sk = []
for i in range(len(s1)):
    tmp = element()
    tmp.spin_init(s1[i])
    tmp1 = element()
    tmp1.spin_init(-np.array(s1[i]))
    sk.append(tmp)
    sk.append(tmp1)
G = group()
G.init(sk)
cl = G.find_class()
print(cl)
#'''
'''
gk = []
for i in range(len(g)):
    tmp = element()
    tmp.init(g[i], s[i])
    tmp1 = element()
    tmp1.init(g[i], -np.array(s[i],dtype='complex'))
    #print("s[i]", s[i])
    gk.append(tmp)
    gk.append(tmp1)

G = group()
G.init(gk)
cl = G.find_class()
print(cl)

H = G.class_mul_constants()
print(H)

character_table = G.burnside_class_table()
print(character_table)
'''

