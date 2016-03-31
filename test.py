import numpy as np
from sim_diag import *

H = [
   np.array( [[1,0,0],[0,1,0],[0,0,1]]), np.array([[0,1,0],[2,1,0],[0,0,2]]), np.array([[0,0,1],[0,0,2],[3,3,0]])
]
#H = np.array(H)
#print(H.shape)
R, L, err = jacobi_angles(np.array( [[1,0,0],[0,1,0],[0,0,1]]), np.array([[0,1,0],[2,1,0],[0,0,2]]), np.array([[0,0,1],[0,0,2],[3,3,0]]))
