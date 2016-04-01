import numpy as np
from numpy import linalg as LA
class element():

    def init(self, a):
        a = np.array(a, dtype='float')
        self.D = a[0:3, 0:3]
        self.t = a[:,3]

    def trans_init(self, a):
        self.D = np.eye(3)
        self.t = np.array(a, dtype='float')
    def element_product(self, a, b):
        self.D = np.dot(a.D , b.D)
        self.t = np.dot(a.D, b.t) + a.t
    def zip_element(self):
        a = np.concatenate((self.D, np.array([self.t]).T), axis = 1)
        return a
    def xyz(self):
        x = str(np.sum(self.D[:, 0])) + 'x' + '+' + str(self.t[0])
        y = str(np.sum(self.D[:, 1])) + 'y' + '+' + str(self.t[1])
        z = str(np.sum(self.D[:, 2])) + 'z' + '+' + str(self.t[2])
        return x + ',' + y + ',' + z

'''
a = element()
b = element()
a.init([[1,0,0,0],[0,-1,0,0.5], [0,0,-1,0]])
b.init([[-1,0,0,0], [0,1,0,0.5], [0,0,-1,0.5]])

c = element()
c.element_product(a,b)
print(c.zip_element())
'''
class group():
    #mtable = None
    def init(self, g, boundary = [1,1,1]):
        self.g = g
        self.boundary = boundary
        self.order = len(self.g)
        self.multi_table()
    def check_equality(self, a, b):
        equal = False
        if (a.D == b.D).all():
            t = a.t - b.t
            if np.mod(t[0], self.boundary[0]) == 0 and np.mod(t[1], self.boundary[1]) == 0 and np.mod(t[2], self.boundary[2]) == 0:
                equal = True
        return equal

    def multi_table(self):
        mtable = []
        #mtable1 = []
        for i in range(self.order):
            line = []
            #line1 = []
            for j in range(self.order):
                tmp = element()

                tmp.element_product(self.g[i], self.g[j])
                for k in range(self.order):
                    if self.check_equality(tmp, self.g[k]) == True:
                        #line.append(tmp)
                        line.append(k)

            #mtable1.append(line1)
            mtable.append(line)

        self.mtable = np.array(mtable, dtype='int')
        return mtable

    def inverse_elelment(self, a): # a is the int number of element
        b = list(self.mtable[a]).index(0)
        return b

    def find_class(self):

        classes = [[0]]
        cnt = 1
        i=1
        while i < self.order and cnt < self.order :
            if i not in [item for sublist in classes for item in sublist]:
                c = [i]
                cnt += 1
                j = 1
                while j < self.order and cnt < self.order :
                    tmp = self.mtable[self.inverse_elelment(j), self.mtable[i, j]]
                    if tmp not in c:
                        c.append(tmp)
                        cnt += 1
                    j += 1
                classes.append(c)
            i += 1
        self.cl = classes
        return classes

    def group_product(self, g1, g2, boundary = [1,2,2]):

        glist = []
        for i in range(g1.order):
            for j in range(g2.order):
                tmp = element()
                tmp.element_product(g1.g[i], g2.g[j])
                #print(tmp.zip_element())
                glist.append(tmp)

        self.init(glist, boundary)

        return self

    def subset_product(self, s1, s2):

        elist = []
        for i in range(len(s1)):
            for j in range(len(s2)):
                tmp = element()
                tmp.element_product(self.g[s1[i]], self.g[s2[j]])
                for k in range(self.order):
                    if self.check_equality(tmp, self.g[k]):
                        elist.append(k)
                        break

        return elist

    def check_class(self, element):
        for i in range(len(self.cl)):
            for j in range(len(self.cl[i])):
                if self.check_equality(element, self.g[self.cl[i][j]]):
                    return i


    def class_mul_constants(self):
        cl = self.cl
        nc = len(cl)
        num_in_cl = [len(i) for i in cl]
        H = np.zeros((nc,nc,nc))
        #H[0,0,0] = 1
        #H[0,1,1] = 1
        #H[0,2,2] = 1
        for i in range(nc):
            for j in range(nc):
                elist = self.subset_product(cl[i],cl[j])
                for k in range(nc):
                    H[i,j,k] = elist.count(cl[k][0])
        self.cl_mat = H
        return H

    def burnside_class_table(self):


        def check_same_vec(vec):
            same_vec_index = []
            for i in range(len(vec)):
                for j in range(i + 1, len(vec)):

                    if np.allclose(vec[i], vec[j]) or np.allclose(vec[i], -vec[j]):
                        same_vec_index.append(j)

            same_vec_index = list(set(same_vec_index))
            same_vec_index = sorted(same_vec_index, reverse = True)

            for i in same_vec_index:
                del vec[i]
            return vec

        #def check_same_eigenval(w):

        def find_nondegenerate_vec(vec, H, nc):
            for i in range(len(H)):
                w,v = LA.eig(H[i])

                w = np.array(list(w))
                w_argsort = np.argsort(w)
                index = []
                for j in range(len(w)):
                    if abs(w[w_argsort[j % len(w)]] - w[w_argsort[(j - 1) % len(w)]]) > 1e-3 and abs(
                                    w[w_argsort[(j + 1) % len(w)]] - w[w_argsort[j % len(w)]]) > 1e-3:
                        index.append(w_argsort[j])
                for j in index:
                    vec.append(v[:,j])
            #print(vec)
            if len(vec) > 1:
                vec = check_same_vec(vec)
            if len(vec) != nc and len(H) > 1:
                h = []
                for i in range(len(H)-1):
                    h.append(H[i] + H[i+1])


                return find_nondegenerate_vec(vec, h, nc)
            elif len(vec) == nc:
                return vec
            else:
                print("ERROR IN RECURSION (burnside method)")


        def normalize_vec(vec):

            for i in range(len(vec)):
                vec[i] = vec[i] * (1/vec[i][0])
            vec = np.array(vec).T
            return vec

        def simul_diag_cl_mat(H, vec):
            nc = len(H)
            cl_table = np.zeros((nc,nc), dtype='complex')
            for i in range(nc):
                diag = np.dot(LA.inv(vec), np.dot(H[i],vec))
                #print("diag", np.diag(diag))
                cl_table[:,i] = np.diag(diag)
            return cl_table

        def get_irrep_dim(cl_table):
            nc = len(cl_table)
            d = []
            for i in range(nc):
                tmp = 0
                for j in range(nc):
                    tmp += cl_table[i,j]* np.conj(cl_table[i,j])/len(self.cl[j])
                d_square = self.order/tmp
                d.append(np.sqrt(d_square))
            return d

        def get_character_table(cl_table,dim):
            nc = len(cl_table)
            character_table = np.zeros((nc, nc),dtype='complex')
            for i in range(nc):
                for j in range(nc):
                    character_table[i,j] = dim[i]/len(cl[j])*cl_table[i,j]
            character_table = character_table[np.argsort(character_table[:, 0])]
            return character_table

        cl = self.cl
        order = self.order
        nc = len(cl)
        vec = []
        H = [i for i in self.cl_mat]
        vec = find_nondegenerate_vec(vec, H, nc)
        #print("before norm", vec)
        vec = normalize_vec(vec)
        #print("after norm",vec)
        cl_table = simul_diag_cl_mat(H, vec)
        #print("cl_table",cl_table)
        dim = get_irrep_dim(cl_table)
        #print("dim", dim)
        character_table = get_character_table(cl_table,dim)
        #print(character_table)
        return character_table


#'''

#@U
g = [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0]], [[-1,0,0,0],[0,-1,0,0],[0,0,1,0.5]],
    [[-1, 0,0,0], [0,1,0,0.5], [0,0,-1,0.5]], [[1,0,0,0],[0,-1,0,0.5],[0,0,-1,0]],
    [[-1,0,0,0], [0,-1,0,0], [0,0,-1,0]], [[1,0,0,0],[0,1,0,0],[0,0,-1,0.5]],
    [[1,0,0,0],[0,-1,0,0.5], [0,0,1,0.5]], [[-1,0,0,0],[0,1,0,0.5], [0,0,1,0]]
]


#@UX 1,2,7,8
#@ZU 1,4,6,8


t = [[0,0,0], [0,1,0]]
gk = []

#x = [i for i in range(len(g))]
x = [0,1,6,7] #UX
#x = [0,2,5,7] #UZ

for i in x:
    tmp = element()
    tmp.init(g[i])
    gk.append(tmp)

tk = []
for i in t:
    tmp = element()
    tmp.trans_init(i)
    tk.append(tmp)

#[print(i.zip_element()) for i in gk]
#[print(i.zip_element()) for i in tk]

Gk = group()
Gk.init(gk)
Tk = group()
Tk.init(tk)
print(Gk.find_class())

G = group()
G.group_product(Gk, Tk, boundary=[1,2,1])
print("g1 element",G.g[1].zip_element())
print(G.order)
print(G.find_class())
print(len(G.find_class()))

H = G.class_mul_constants()
#print(H)

character_table = G.burnside_class_table()
np.set_printoptions(precision=3)
np.savetxt('ct', character_table, '%5.2f')

#'''