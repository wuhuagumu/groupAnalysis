import numpy as np
from numpy import linalg as LA


def p2r(radii, angles):
    return radii * np.exp(1j*angles)


def r2p(x):
    return abs(x), np.angle(x)


def gram_schmidt(vec):
    result = []
    dim = len(vec[0])
    for i in range(len(vec)):
        tmp = np.zeros(dim, dtype='complex')
        for j in range(i):
            tmp += - np.dot(np.conj(vec[i]), result[j]) / (np.dot(np.conj(result[j]), result[j])) * vec[j]
        result.append(tmp + vec[i])

    for i in range(len(result)):
        result[i] = result[i] / LA.norm(result[i])
    return result


class element():
    # the element class contains three variables: D, t, s standing for rotation, translation, and spin-1/2 operation
    # In the element class, we defined three init method, including
    # init(spacial operation, spinor operation):
    # trans_init(spaceal operation) , spinor is defined to [[1,0],[0,1] ]
    # spin_init(spinor operation), spacial is defined to [[1,0,0,0],[0,1,0,0],[0,0,0,1]]
    # rotation_init(rotation operation), translation is [0,0,0], spinor is [[1,0],[0,1]]
    # check equality(), check whether two element is equivalent, returns boolean
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
    def init(self, a, s=None, dim = 3):
        a = np.array(a, dtype='float')
        self.D = a[0:dim, 0:dim]
        self.t = a[:, dim]
        if s is None:
            self.s = np.eye(2, dtype='complex')
        else:
            self.s = np.array(s, dtype='complex')

    def trans_init(self, a, dim=3):
        self.D = np.eye(dim)
        self.t = np.array(a, dtype='float')
        self.s = np.eye(2, dtype='complex')

    def spin_init(self, s, dim=3):
        self.D = np.eye(dim)
        self.t = np.zeros(dim)
        self.s = np.array(s, dtype='complex')

    def rotation_init(self, a, dim=3):
        self.D = a
        self.t = np.zeros(dim, dtype='float')
        self.s = np.eye(2, dtype='complex')

    def element_product(self, a, b):
        self.D = np.dot(a.D, b.D)
        self.t = np.dot(a.D, b.t) + a.t
        self.s = np.dot(a.s, b.s)

    def check_equality(self, a, b, kpt):
        equal = False
        if np.allclose(a.D, b.D) and np.allclose(a.s, b.s):
            t = a.t - b.t
            #print(np.dot(t, kpt))
            if np.allclose(t, 0) == True:
                equal = True
            elif np.allclose(np.mod(np.dot(t, kpt), 1), 0):
                equal = True
        return equal

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
    # multi_table(), get multiplication table of this group
    # iverse_element(a), from multiplication table, get the iverse of a. a is the number of the element in g
    # find_class(), classify the group element by conjugation, return cl (short for class)
    # group_product(g1, g2), multiply of two group and init the multiplication as a third group, return self
    # subset_product(s1, s2), multiplication of two subset of this group, used in calculating class multiplication
    # check_class(a), check which class element a belongs
    # class_mul_constants(), find class multiplication constants used for calculating charactertable, return cl_mat
    # burnside_class_table(), burnside method for getting character table, return character_table
    # the last part is about how to get irrep
    # example usage is at the end

    def init(self, g, kpt=[0, 0, 0]):
        self.g = g
        self.kpt = kpt
        self.order = len(self.g)
        self.multi_table()

    def multi_table(self):
        mtable = []
        for i in range(self.order):
            line = []
            for j in range(self.order):
                tmp = element()
                tmp.element_product(self.g[i], self.g[j])
                for k in range(self.order):
                    if tmp.check_equality(tmp, self.g[k], self.kpt) == True:
                        line.append(k)
                        break
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

    def group_product(self, g1, g2, kpt=[0, 0, 0]):
        glist = []
        for i in range(g1.order):
            for j in range(g2.order):
                tmp = element()
                tmp.element_product(g1.g[i], g2.g[j])
                glist.append(tmp)
        self.init(glist, kpt)
        return self

    def subset_product(self, s1, s2):
        elist = []
        for i in range(len(s1)):
            for j in range(len(s2)):
                tmp = element()
                tmp.element_product(self.g[s1[i]], self.g[s2[j]])
                for k in range(self.order):
                    if tmp.check_equality(tmp, self.g[k], self.kpt):
                        elist.append(k)
                        break
        return elist

    def check_class(self, element):
        for i in range(len(self.cl)):
            for j in range(len(self.cl[i])):
                if element.check_equality(element, self.g[self.cl[i][j]], self.kpt):
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
                for j in range(len(vec[i])):
                    if not np.allclose(vec[i][j], 0):
                        flag = j
                        break

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
                        if abs(w[j] - w[k]) < 1e-5:
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

    def regular_rep(self, element):  # element is the number of the element in group
        reg_rep = np.zeros((self.order, self.order))
        for i in range(self.order):
            reg_rep[self.mtable[element, i], i] = 1
        return reg_rep

    def element_order(self, element):
        i = 0
        power = 0
        element_order = None
        while i < self.order:
            i += 1
            power = self.mtable[power, element]
            if power == 0:
                element_order = i
                break
        if element_order is None:
            print("ERROR:, can not find element order")
            return
        return element_order

    def reg_eigencolumns(self, reg_rep):
        x = range(self.order)
        y = np.dot(reg_rep, x)
        y = np.array(y, dtype='int')
        # print("reg_rep \n",reg_rep)
        print("after rearrangement", y)
        permutation_cycle = []
        # determine cycle and cycle length
        i = 0
        while i < self.order:
            if i not in [item for sublist in permutation_cycle for item in sublist]:
                j = i
                cnt = 1
                # this count num has no use actually
                l = [i]
                while y[j] != x[i] and cnt <= self.order:
                    cnt += 1
                    j = x[y[j]]
                    l.append(j)
                permutation_cycle.append(l)
            i += 1
        print("permutation cycle: ", permutation_cycle)

        # permutation cycle determined, the next is to determine the eigenvector
        eigencolumns = np.zeros((self.order, self.order), dtype='complex')
        cnt = 0
        eigenvalues = np.zeros(self.order, dtype='complex')
        for i in range(len(permutation_cycle)):
            length = len(permutation_cycle[i])
            tmp = permutation_cycle[i]
            for j in range(length):
                powers = p2r(1, 2 * np.pi / length * j)

                eigencolumns[tmp[0], cnt] = 1

                for k in range(length - 1):
                    eigencolumns[tmp[k + 1], cnt] = pow(powers, (k + 1))
                eigenvalues[cnt] = eigencolumns[permutation_cycle[i][1], cnt]
                cnt += 1

        return eigenvalues, eigencolumns

    def projection_operator(self, irrep_index, character_table, reg_rep):
        # irrep_index is the index of the irrep in character table
        dim = character_table[irrep_index, 0]
        projection_operator = np.zeros((self.order, self.order), dtype='complex')
        for i in range(len(self.cl)):
            for j in range(len(self.cl[i])):
                projection_operator += np.conj(character_table[irrep_index, i]) * reg_rep[self.cl[i][j]]
        projection_operator = dim / self.order * projection_operator
        return projection_operator

    def vec_same(self, vec1, vec2):
        flag = None
        for i in range(len(vec1)):
            if not np.allclose(vec1[i], 0):
                flag = i
        if not np.allclose(vec2[flag], 0):
            if np.allclose(vec1, (vec1[flag] / vec2[flag]) * vec2):
                return True
        return False

    def check_same_vec(self, vec):
        same_vec_index = []
        for i in range(len(vec)):
            for j in range(len(vec[i])):
                if not np.allclose(vec[i][j], 0):
                    flag = j

            for j in range(i + 1, len(vec)):
                if not np.allclose(vec[j][flag], 0):
                    if np.allclose(vec[i], (vec[i][flag] / vec[j][flag]) * vec[j]):
                        same_vec_index.append(j)

        same_vec_index = list(set(same_vec_index))
        same_vec_index = sorted(same_vec_index, reverse=True)

        for i in same_vec_index:
            del vec[i]
        return vec

    def subspace_eigenvector(self, irrep_index, character_table, reg_rep):
        projection_operator = self.projection_operator(irrep_index, character_table, reg_rep)
        dim = round(abs(character_table[irrep_index, 0]))
        np.savetxt('proj_opera', projection_operator, '%5.2f')

        # for we always choose reg_rep[1] as a start point
        for i in range(1, self.order):
            eigenvalues, eigencolumns = self.reg_eigencolumns(reg_rep[i])
            projected_vector = np.dot(projection_operator, eigencolumns)
            np.savetxt('eigencolumns', eigencolumns, '%5.2f')
            np.savetxt('projected', projected_vector, '%5.2f')

            # find those eigencolumns that does not change under projection,
            # save them and their corresponding eigenvalues of reg_rep[i]

            vec = []
            for j in range(self.order):
                if not np.allclose(projected_vector[:, j], 0):
                    vec.append(projected_vector[:, j])

            vec = self.check_same_vec(vec)
            l = len(vec)

            vec = np.array(vec, dtype='complex').T
            eigenvector = np.dot(reg_rep[i], vec)

            # find non-degenerate subspace vector
            lam = []
            for j in range(l):
                if self.vec_same(eigenvector[:, j], vec[:, j]):
                    for k in range(self.order):
                        if not np.allclose(vec[k, j], 0):
                            lam.append(eigenvector[k, j] / vec[k, j])
                            break
            print("lam", lam)
            # check the projected eigencolumns are not degenerate,
            # if degenerate, start from a different reg_rep[i]
            # the standard is whether same eigenvalues appeared more than dim times
            # first count the occurences of eigenvals of those eigencolumns that survived the projection
            if len(vec) != 0:
                same_val = []
                count = []

                for k in range(len(lam)):
                    cnt = 0
                    tmp = []
                    for j in range(k, len(lam)):
                        if abs(lam[j] - lam[k]) < 1e-5 and \
                                (j not in [item for sublist in same_val for item in sublist]):
                            cnt += 1
                            tmp.append(j)
                    count.append(cnt)
                    same_val.append(tmp)
                print("count", count, "same val", same_val)
                if all(j <= dim for j in count):
                    # choose different starting eigenvector might result in different irrep, but can be related
                    # by a unitary transformation
                    flag = 0
                    # at this time dim must not equal to 1
                    sub_vec = [vec[:, flag]]
                    for j in range(1, self.order):
                        if j != i:
                            new_vec = np.dot(reg_rep[j], vec[:, flag])
                            sub_vec.append(new_vec)
                    sub_vec = self.check_same_vec(sub_vec)
                    if len(sub_vec) != dim:
                        print("Error, subspace vector is wrong")
                        return

                    # orthonormalization
                    sub_vec = gram_schmidt(sub_vec)
                    sub_vec = np.array(sub_vec, dtype='complex').T

                    return sub_vec

    def irrep(self, irrep_index):
        # note that the dim of this irrep_index should be >1
        # if the dim ==1, return character table
        print("self mtable", self.mtable)
        ctable = self.burnside_character_table()
        print("character table \n", ctable)
        dim = ctable[irrep_index, 0]
        print("dim", dim)
        if np.allclose(ctable[irrep_index, 0], 1):
            return ctable[irrep_index]

        reg_rep = np.zeros((self.order, self.order, self.order), dtype='int')
        for i in range(self.order):
            reg_rep[i, :, :] = self.regular_rep(i)
        vec = self.subspace_eigenvector(irrep_index, ctable, reg_rep)

        irrep = []
        for i in range(self.order):
            tmp = np.dot(np.conj(vec.T), np.dot(reg_rep[i], vec))
            irrep.append(tmp)
        return irrep


# '''

g = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.5]],
    [[-1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, -1, 0.5]], [[1, 0, 0, 0], [0, -1, 0, 0.5], [0, 0, -1, 0]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.5]],
    [[1, 0, 0, 0], [0, -1, 0, -0.5], [0, 0, 1, -0.5]], [[-1, 0, 0, 0], [0, 1, 0, -0.5], [0, 0, 1, 0]]
]

s = [
    [[1, 0], [0, 1]], [[-1j, 0], [0, 1j]], [[0, -1], [1, 0]], [[0, -1j], [-1j, 0]],
    [[1, 0], [0, 1]], [[-1j, 0], [0, 1j]], [[0, -1], [1, 0]], [[0, -1j], [-1j, 0]]
]

t = [[0, 0, 0], [0, 1, 0]]


x = [i for i in range(len(g))]
#x = [0, 1, 6, 7]  # UX
# x = [0,2,5,7] #UZ

gk = []
for i in x:
    tmp = element()
    tmp1 = element()
    tmp.init(g[i], s[i])
    tmp1.init(g[i], -np.array(s[i], dtype='complex'))
    gk.append(tmp)
    gk.append(tmp1)


tk = []
for i in t:
    tmp = element()
    tmp.trans_init(i)
    tk.append(tmp)

Gk = group()
Gk.init(gk)
Tk = group()
Tk.init(tk)
print(Gk.find_class())

G = group()
G.group_product(Tk, Gk, kpt=[0,0.5,0.5])
print("order", G.order)
print("class", G.find_class())
print("nc", len(G.find_class()))

H = G.class_mul_constants()

character_table = G.burnside_character_table()
np.set_printoptions(precision=3)
np.savetxt('ctd', character_table, '%5.2f')

irrep = G.irrep(9)
print("irrep\n", irrep)


file = open('irrep-dg', 'w')

g_irrep = []
for i in range(len(irrep)):

    tmp = G.g[i]
    tmp1 = element()
    tmp1.rotation_init(irrep[i], dim=2)
    g_irrep.append(tmp1)
    print(i,file=file)
    print(tmp.zip_element(),file=file)
    print(irrep[i],file=file)

file.close()

G_irrep = group()
G_irrep.init(g_irrep)
print("G_irrep multiply table\n", G_irrep.mtable)

g_reg = []
for i in range(G.order):
    tmp = element()
    tmp.rotation_init(G.regular_rep(i), dim=32)
    g_reg.append(tmp)

G_reg = group()
G_reg.init(g_reg, kpt = np.zeros(32))
print("G_reg multiply table is same as G.mtable?\n", np.allclose(G_reg.mtable, G.mtable))