import numpy as np
from numpy import linalg as LA


class element():
    # the element class contains three variables: D, t, s standing for rotation, translation, and spin-1/2 operation
    # In the element class, we defined three init method, including
    # init(spacial operation, spinor operation):
    # trans_init(spaceal operation) , spinor is defined to [[1,0],[0,1] ]
    # spin_init(spinor operation), spacial is defined to [[1,0,0,0],[0,1,0,0],[0,0,0,1]]
    # usage example:
    # for i in x:
    #     tmp = element()
    #     tmp1 = element()
    #     tmp.init(g[i], s[i])
    #     tmp1.init(g[i], -np.array(s[i], dtype='complex'))
    #     gk.append(tmp)
    #     gk.append(tmp1)
    #
    # we also defined several element operations, like:
    # element_product(element_1, element_2),
    #     note that product is defined by operate successively of a on b, spacial and spinor seperately
    # zip_element() is used to show spacial operation tightly
    # xyz() is under development
    def init(self, a, s=None):
        a = np.array(a, dtype='float')
        self.D = a[0:3, 0:3]
        self.t = a[:, 3]
        if s is None:
            self.s = np.eye(2, dtype='complex')
        else:
            self.s = np.array(s, dtype='complex')

    def trans_init(self, a):
        self.D = np.eye(3)
        self.t = np.array(a, dtype='float')
        self.s = np.eye(2, dtype='complex')

    def spin_init(self, s):
        self.D = np.eye(3)
        self.t = np.zeros(3)
        self.s = np.array(s, dtype='complex')

    def element_product(self, a, b):
        self.D = np.dot(a.D, b.D)
        self.t = np.dot(a.D, b.t) + a.t
        self.s = np.dot(a.s, b.s)

    def zip_element(self):
        a = np.concatenate((self.D, np.array([self.t]).T), axis=1)
        s = self.s
        return a, s

    def xyz(self):
        x = str(np.sum(self.D[:, 0])) + 'x' + '+' + str(self.t[0])
        y = str(np.sum(self.D[:, 1])) + 'y' + '+' + str(self.t[1])
        z = str(np.sum(self.D[:, 2])) + 'z' + '+' + str(self.t[2])
        return x + ',' + y + ',' + z


class group():
    # Group mainly deals with space group and double space group. group element is encapsulated in class element
    # group class contains variable variables, like:
    #   g, all group element in list
    #   boundary, used when checking equal of element, this variable has default [1,1,1], and other values are used when
    #       we encountered non-symmorphic symmetries
    #   order, num of element in g
    #   mtable, multiply table of this group, only index of the element is stored and used
    # all variables above are inited by init method and exist in any group.
    # optional variables are:
    #   cl, classes, stored in list and obtained by find_class() method by definition
    #   cl_mat, short for class matrix, meaning the class multiplication constants defined by class multiplication
    #       obtained cl_mat in class_mul_constants()
    #
    # function list:
    # check equality(), check whether two element is equivalent, returns boolean
    # multi_table(), get multiplication table of this group
    # iverse_element(a), from multiplication table, get the iverse of a. a is the number of the element in g
    # find_class(), classify the group element by conjugation, return cl (short for class)
    # group_product(g1, g2), multiply of two group and init the multiplication as a third group, return self
    # subset_product(s1, s2), multiplication of two subset of this group, used in calculating class multiplication
    # check_class(a), check which class element a belongs
    # class_mul_constants(), find class multiplication constants used for calculating charactertable, return cl_mat
    # burnside_class_table(), burnside method for getting character table, return character_table
    # example usage is at the end

    def init(self, g, boundary=[1, 1, 1]):
        self.g = g
        self.boundary = boundary
        self.order = len(self.g)
        self.multi_table()

    def check_equality(self, a, b):
        equal = False
        if np.allclose(a.D, b.D) and np.allclose(a.s, b.s):
            t = a.t - b.t
            if np.mod(t[0], self.boundary[0]) == 0 and np.mod(t[1], self.boundary[1]) == 0 and \
                            np.mod(t[2], self.boundary[2]) == 0:
                equal = True
        return equal

    def multi_table(self):
        mtable = []
        for i in range(self.order):
            line = []
            for j in range(self.order):
                tmp = element()
                tmp.element_product(self.g[i], self.g[j])
                for k in range(self.order):
                    if self.check_equality(tmp, self.g[k]) == True:
                        line.append(k)
            mtable.append(line)
        self.mtable = np.array(mtable, dtype='int')
        return mtable

    def inverse_elelment(self, a):
        # a is the int number of element
        b = list(self.mtable[a]).index(0)
        return b

    def find_class(self):
        classes = [[0]]
        cnt = 1
        i = 1
        while i < self.order and cnt < self.order:
            if i not in [item for sublist in classes for item in sublist]:
                c = [i]
                cnt += 1
                j = 1
                while j < self.order and cnt < self.order:
                    tmp = self.mtable[self.inverse_elelment(j), self.mtable[i, j]]
                    if tmp not in c:
                        c.append(tmp)
                        cnt += 1
                    j += 1
                classes.append(c)
            i += 1
        self.cl = classes
        return classes

    def group_product(self, g1, g2, boundary=[1, 2, 2]):
        glist = []
        for i in range(g1.order):
            for j in range(g2.order):
                tmp = element()
                tmp.element_product(g1.g[i], g2.g[j])
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
        # num_in_cl = [len(i) for i in cl]
        H = np.zeros((nc, nc, nc))
        for i in range(nc):
            for j in range(nc):
                elist = self.subset_product(cl[i], cl[j])
                for k in range(nc):
                    H[i, j, k] = elist.count(cl[k][0])
        self.cl_mat = H
        return H

    def burnside_character_table(self):
        # burnside method, try to find vectors that can simultaneously diagonalize cl_mat
        # the method is to find all non-degenerate eigenvectors, if not enough (we need number of class vectors)
        # add H[i] + H[j] iteratively
        # check_same_vec(vec_list) is used to check whether the obtained two vectors are same up to a phase factor
        # find_nondegenerate_vec() searches all H matrices to get vectors with non-equivalent eigenvalues
        # normalize_vec(), to normalize every vector we found to get the correct class constants
        # simul_diag_cl_mat(H, vec), after diag every H matrix, the diagonal elements are the class constants
        #   returns cl_table
        # get_irrep_dim(cl_table), calculating every irrep's dimension
        # get_character_table(cl_table,dim), get character table from cl_table and dim
        def check_same_vec(vec):
            same_vec_index = []
            for i in range(len(vec)):
                x = np.nonzero(vec[i])
                flag = x[0][0]
                for j in range(i + 1, len(vec)):
                    if not np.allclose(vec[j][flag], 0):
                        if np.allclose(vec[i], (vec[i][flag] / vec[j][flag]) * vec[j]):
                            same_vec_index.append(j)

            same_vec_index = list(set(same_vec_index))
            same_vec_index = sorted(same_vec_index, reverse=True)

            for i in same_vec_index:
                del vec[i]
            return vec

        # def check_same_eigenval(w):

        def find_nondegenerate_vec(vec, H, nc, index_w):
            # index_w exists for checking when wrong, this method might be problematic as it contains round off errors
            for i in range(len(H)):
                w, v = LA.eig(H[i])
                w = np.array(list(w))
                index = []
                w1 = []
                # fingding non-equivalent eigenvalues
                for k in range(len(w)):
                    cnt = 0
                    for j in range(len(w)):
                        if abs(w[j] - w[k]) < 1e-3:
                            cnt += 1
                    if cnt == 1:
                        index.append(k)
                # end finding non-equivalent eigenvalues, and its index stored in k

                # start picking out those non-degenerate vecs
                for j in index:
                    vec.append(v[:, j])
                    w1.append(w[j])
                index_w.append(w1)

            if len(vec) > 1:
                vec = check_same_vec(vec)
            if len(vec) != nc and len(H) > 1:
                h = []
                for i in range(len(H) - 1):
                    h.append(H[i] + H[i + 1])

                return find_nondegenerate_vec(vec, h, nc, index_w)
            elif len(vec) == nc:
                return vec
            else:
                # need debugging when here, the main reason is the number of vec found early is not compatible with
                # num of class
                # print("vec", np.shape(vec))
                print("index_w", index_w)
                np.savetxt('vec', np.array(vec, dtype='complex'), '%5.2f')
                print("ERROR IN RECURSION (burnside method), vector num != number of classes")
                print("vec shape", len(vec))
                return vec

        def normalize_vec(vec):
            for i in range(len(vec)):
                vec[i] = vec[i] * (1 / vec[i][0])
            vec = np.array(vec).T
            return vec

        def simul_diag_cl_mat(H, vec):
            nc = len(H)
            cl_table = np.zeros((nc, nc), dtype='complex')
            for i in range(nc):
                diag = np.dot(LA.inv(vec), np.dot(H[i], vec))
                cl_table[:, i] = np.diag(diag)
            return cl_table

        def get_irrep_dim(cl_table):
            nc = len(cl_table)
            d = []
            for i in range(nc):
                tmp = 0
                for j in range(nc):
                    tmp += cl_table[i, j] * np.conj(cl_table[i, j]) / len(self.cl[j])
                d_square = self.order / tmp
                d.append(np.sqrt(d_square))
            return d

        def get_character_table(cl_table, dim):
            nc = len(cl_table)
            character_table = np.zeros((nc, nc), dtype='complex')
            for i in range(nc):
                for j in range(nc):
                    character_table[i, j] = dim[i] / len(cl[j]) * cl_table[i, j]
            character_table = character_table[np.argsort(character_table[:, 0])]
            return character_table

        # follows is the procedure for getting charactor table
        # 1. find_nondegenerate_vec with eigenvalue exist once in every H matrix
        # (find-check-find-check) until the num of vecs equals number of classes
        # 2. normalize_vec
        # 3. simul_diag_cl_mat by vec to get cl constants
        # 4. get dimension of irreps
        # 5. get character table using cl constants and dim

        cl = self.cl
        order = self.order
        nc = len(cl)
        vec = []
        index_w = []
        H = [i for i in self.cl_mat]
        vec = find_nondegenerate_vec(vec, H, nc, index_w)
        vec = normalize_vec(vec)
        cl_table = simul_diag_cl_mat(H, vec)
        dim = get_irrep_dim(cl_table)
        character_table = get_character_table(cl_table, dim)
        return character_table


# '''

g = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.5]],
    [[-1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, -1, 0.5]], [[1, 0, 0, 0], [0, -1, 0, 0.5], [0, 0, -1, 0]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0.5]],
    [[1, 0, 0, 0], [0, -1, 0, 0.5], [0, 0, 1, 0.5]], [[-1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0]]
]

s = [
    [[1, 0], [0, 1]], [[-1j, 0], [0, 1j]], [[0, -1], [1, 0]], [[0, -1j], [-1j, 0]],
    [[1, 0], [0, 1]], [[-1j, 0], [0, 1j]], [[0, -1], [1, 0]], [[0, -1j], [-1j, 0]]
]

t = [[0, 0, 0], [0, 1, 0]]
gk = []

x = [i for i in range(len(g))]
#x = [0, 1, 6, 7]  # UX
# x = [0,2,5,7] #UZ

for i in x:
    tmp = element()
    tmp1 = element()
    tmp.init(g[i], s[i])
    tmp1.init(g[i], -np.array(s[i], dtype='complex'))
    gk.append(tmp)
    gk.append(tmp1)

print("gk len", len(gk))
tk = []
for i in t:
    tmp = element()
    tmp.trans_init(i)
    tk.append(tmp)

# [print(i.zip_element()) for i in gk]
# [print(i.zip_element()) for i in tk]

Gk = group()
Gk.init(gk)
Tk = group()
Tk.init(tk)
print(Gk.find_class())

G = group()
G.group_product(Gk, Tk, boundary=[1, 2, 1])
print("order", G.order)
print("class", G.find_class())
print("nc", len(G.find_class()))

for i in range(4):
    tmp = G.g[i]
    print(i, tmp.zip_element())
H = G.class_mul_constants()
# print(H)

character_table = G.burnside_character_table()
np.savetxt('ctd', character_table, '%5.2f')
# '''
