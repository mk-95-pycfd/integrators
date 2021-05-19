import sympy as sp
from sympy import IndexedBase,Idx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import  time
from Jacobian_indexing import PeriodicIndexer
import matplotlib.pyplot as plt

class Analytic_Jacobian:
    def __init__(self,resifual_functions,diff_vars,indicies,stencil,stencil_pts,prob_description,indexer,is_scalar=False):
        self.resid_funcs = resifual_functions
        self.diff_vars = diff_vars
        self.stencil = stencil # todo: make sure to remove any dependence on stencil and replace it with pairs
        self.stencil_pts = stencil_pts # ex: [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
        self.prob_description = prob_description
        self.j, self.i = indicies
        self.Indexer = indexer
        self.is_scalar =is_scalar
        self.diffs = self.__diffs_f_X()



    def __sym_diff_f_x_pts(self,f,X):
        '''
        Computes the symbolic derivatives of the function f with respect to the sencil points X[j+pt_j, i + pt_i] for the all points (j,i)
        :param f: residual function in symbolic version
        :param X: symbolic independent variable
        :return: dictionary with the lamdified functions of the derivatives with the keys being the stencil point (pt_j, pt_i)
        '''
        diffs = {}
        for pt_j, pt_i in self.stencil_pts:
            sym_diff = f.diff(X[self.j + pt_j, self.i + pt_i])
            diff_lam = sp.lambdify([sp.DeferredVector('unp1'), sp.DeferredVector('vnp1'), self.j, self.i], sym_diff,'numpy')
            diffs[(pt_j, pt_i)] = diff_lam
        return diffs

    def __diffs_f_X(self):
        '''
        generates lambda function for the derivatives of all residual functions f with resprect to all X
        :return: dict with all the generated lambda functions with keys 'f{i}_X{j}' i is the number of the function in the list
        of residual functions and j is the number of the independent variables in the list of independnet vars.
        '''
        diffs = {}
        tic = time.perf_counter()
        for i in range(1,len(self.resid_funcs)+1):
            for j in range(1,len(self.diff_vars)+1):
                key = 'f{}_X{}'.format(i,j)
                diffs[key] = self.__sym_diff_f_x_pts(self.resid_funcs[i-1], self.diff_vars[j-1])
        toc = time.perf_counter()
        lamdify_diffs_time = toc - tic
        print('lamdify_diff_time=', lamdify_diffs_time)
        return diffs

    def Sparse_Jacobian(self, ukp1, vkp1=None, pkp1=None):
        '''Compute the sparce jacobian'''

        if not self.is_scalar:
            ukp1, vkp1 = self.augment_fields_size_NSE(ukp1, vkp1)

        # plt.imshow(ukp1, origin='bottom')
        # plt.show()
        #
        # plt.imshow(vkp1, origin='bottom')
        # plt.show()

        nx = self.prob_description.nx
        ny = self.prob_description.ny

        size = (nx) * (ny)
        block_Jacobian ={}
        for j in range(1,len(self.resid_funcs)+1):
            for i in range(1,len(self.diff_vars)+1):
                key = 'df{}dx{}'.format(j,i)
                block_Jacobian[key] = lil_matrix(np.zeros([size, size]))

        lam_eval_time = 0
        idxer = self.Indexer(nx, ny)
        for index_j in range(1, ny + 1):
            for index_i in range(1, nx + 1):
                row = idxer.linear_indexer_row(index_j, index_i)  # row in the jacobian matrix
                local_idxer_column = idxer.linear_indexer_column(index_j,index_i)  # return a function that takes a stencil point and return column in Jacobian matrix
                for J, I in self.stencil_pts:
                    column = local_idxer_column(J, I)

                    tic = time.perf_counter()
                    for fj in range(1, len(self.resid_funcs) + 1):
                        for xi in range(1, len(self.diff_vars) + 1):
                            key = 'df{}dx{}'.format(fj, xi)
                            diff_key = 'f{}_X{}'.format(fj, xi)
                            block_Jacobian[key][row, column] += self.diffs[diff_key][(J, I)](ukp1, vkp1, index_j,index_i)

                    toc = time.perf_counter()
                    lam_eval_time += toc - tic

        # create the entire sparce Jacobian matrix
        Rows =[]
        # assembling the row blocks
        for fj in range(1, len(self.resid_funcs) + 1):
            row =[]
            for xi in range(1, len(self.diff_vars) + 1):
                key = 'df{}dx{}'.format(fj, xi)
                row.append(block_Jacobian[key].tocsr())
            Rows.append(hstack(row))
        # assembling the entire matrix
        J = vstack(Rows)

        print('lam_eval_time=', lam_eval_time)
        return J.tocsr()

    def augment_fields_size_NSE(self,field_x,field_y):
        # add an extra ghost cell in the x direction and apply periodic boundary condition on field_x
        # add an extra ghost cell in the y direction and apply periodic boundary condition on field_y
        nx = self.prob_description.nx
        ny = self.prob_description.ny

        field_x_size_x,field_x_size_y=field_x.shape
        field_y_size_x,field_y_size_y=field_y.shape

        new_field_x = np.pad(field_x,((0,1),(0,1)), mode='constant', constant_values=0)
        new_field_y = np.pad(field_y,((0,1),(0,1)), mode='constant', constant_values=0)

        new_field_x[field_x_size_y,1:nx+1] = field_x[2,1:nx+1]
        new_field_x[1:ny+1,field_x_size_x] = field_x[1:nx+1,2]
        new_field_x[ny+1:,nx+1:] = field_x[1:3,1:3]
        new_field_x[0,nx+1] = field_x[ny,1]
        new_field_x[ny+1:,0] = field_x[1,nx]

        new_field_y[field_y_size_y, 1:nx + 1] = field_y[2, 1:nx + 1]
        new_field_y[1:ny + 2, field_y_size_x] = field_y[1:nx + 2, 2]
        new_field_y[ny + 1:, nx + 1:] = field_y[1:3, 1:3]
        new_field_y[0, nx + 1:] = field_y[ny, 1]
        new_field_y[ny + 1:, 0] = field_y[1, nx]

        return new_field_x, new_field_y

    #---------------------------------------------------------------
    # the following functions are deprecated I am keeping them only
    # for reference.
    #---------------------------------------------------------------
    def __sym_diff_f_x(self,f,X,name):
        """ This function is deprecated"""
        diffs = {}
        # for pt_j in range(stencil[0], stencil[1] + 2): # plus two because the stencile is from -1 to +2
        #     for pt_i in range(stencil[0], stencil[1] + 2): # plus two because the stencile is from -1 to +2
        for pt_j, pt_i in [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]:
                # diffs[(pt_j,pt_i)] = f.diff(X[j+pt_j, i + pt_i])
                # diff_lam = sp.lambdify([sp.DeferredVector('unp1'), sp.DeferredVector('vnp1'), self.j, self.i],
                #                        f.diff(X[ self.j + pt_j, self.i + pt_i]), 'numpy')
                sym_diff=f.diff(X[self.j + pt_j, self.i + pt_i])
                print("{}, point ({},{}): {}".format(name,pt_j,pt_i,sym_diff))
                diff_lam = sp.lambdify([sp.DeferredVector('unp1'), sp.DeferredVector('vnp1'), self.j, self.i],sym_diff, 'numpy')
                diffs[(pt_j, pt_i)] = diff_lam
        return diffs

    def __sym_diff_f_x_scalar(self,f,X,stencil):
        """ This function is deprecated"""
        diffs = {}
        for pt_j in range(stencil[0], stencil[1] + 1):
            for pt_i in range(stencil[0], stencil[1] + 1):
                # diffs[(pt_j,pt_i)] = f.diff(X[j+pt_j, i + pt_i])
                diff_lam = sp.lambdify([sp.DeferredVector('phi_np1'), self.j, self.i],
                                       f.diff(X[ self.j + pt_j, self.i + pt_i]), 'numpy')
                diffs[(pt_j, pt_i)] = diff_lam
        return diffs

    def __diffs_f_X_deprecated(self):
        """ This function is deprecated"""
        f1, f2, f3 = self.resid_funcs
        x1, x2, x3 = self.diff_vars
        stencil = self.stencil
        diffs={}
        tic = time.perf_counter()
        print('f1: ', f1)
        print('f2: ', f2)
        print('f3: ', f3)
        diffs['f1_X1'] = self.__sym_diff_f_x(f1, x1, 'f1_X1')
        diffs['f2_X1'] = self.__sym_diff_f_x(f2, x1, 'f2_X1')
        diffs['f3_X1'] = self.__sym_diff_f_x(f3, x1, 'f3_X1')

        diffs['f1_X2'] = self.__sym_diff_f_x(f1, x2, 'f1_X2')
        diffs['f2_X2'] = self.__sym_diff_f_x(f2, x2, 'f2_X2')
        diffs['f3_X2'] = self.__sym_diff_f_x(f3, x2, 'f3_X2')

        diffs['f1_X3'] = self.__sym_diff_f_x(f1, x3, 'f1_X3')
        diffs['f2_X3'] = self.__sym_diff_f_x(f2, x3, 'f2_X3')
        diffs['f3_X3'] = self.__sym_diff_f_x(f3, x3, 'f3_X3')

        toc = time.perf_counter()
        lamdify_diffs_time = toc - tic
        print('lamdify_diff_time=', lamdify_diffs_time)
        return diffs

    def __diffs_f_X_scalar(self):
        """ This function is deprecated"""
        f = self.resid_funcs
        x = self.diff_vars
        stencil = self.stencil
        diffs={}
        tic = time.perf_counter()
        diffs['f_X'] = self.__sym_diff_f_x_scalar(f, x, stencil)
        toc = time.perf_counter()
        lamdify_diffs_time = toc - tic
        print('lamdify_diff_time=', lamdify_diffs_time)
        return diffs

    def __diffs_f_X_2_resid_func(self):
        """ This function is deprecated"""
        f1,f2 = self.resid_funcs
        x1,x2 = self.diff_vars
        stencil = self.stencil
        diffs={}
        tic = time.perf_counter()
        diffs['f1_X1'] = self.__sym_diff_f_x(f1, x1, stencil)
        diffs['f2_X1'] = self.__sym_diff_f_x(f2, x1, stencil)

        diffs['f1_X2'] = self.__sym_diff_f_x(f1, x2, stencil)
        diffs['f2_X2'] = self.__sym_diff_f_x(f2, x2, stencil)
        toc = time.perf_counter()
        lamdify_diffs_time = toc - tic
        print('lamdify_diff_time=', lamdify_diffs_time)
        return diffs

    def Sparse_Jacobian_2_Resid(self,unp1,vnp1):
        """ This function is deprecated"""
        # numpy_unp1,numpy_vnp1 = self.__augment_fields_size(unp1,vnp1)
        numpy_unp1,numpy_vnp1 = self.augment_fields_size_NSE(unp1,vnp1)
        nx = self.prob_description.nx
        ny = self.prob_description.ny

        size = (nx) * (ny)
        # dfdx = csr_matrix(np.zeros([size,size]))
        df1dx1 = lil_matrix(np.zeros([size, size]))
        df2dx1 = lil_matrix(np.zeros([size, size]))
        df1dx2 = lil_matrix(np.zeros([size, size]))
        df2dx2 = lil_matrix(np.zeros([size, size]))

        lam_eval_time = 0
        idxer = self.Indexer(nx, ny)
        for index_j in range(1, ny + 1):
            for index_i in range(1, nx + 1):
                row = idxer.linear_indexer_row(index_j, index_i)  # row in the jacobian matrix
                local_idxer_column = idxer.linear_indexer_column(index_j,index_i)  # return a function that takes a stencil point and return column in Jacobian matrix
                for J, I in [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]:
                # for J, I in [(-1, 0),(-1, 1), (0, -1), (0, 0), (0, 1),(0, 2),(1, -1),(1, 0),(1, 1),(2, 0)]:
                # for J in range(-1, 1 + 2): # plus two because the stencile is from -1 to +2
                #     for I in range(-1, 1 + 2): # plus two because the stencile is from -1 to +2
                        column = local_idxer_column(J, I)

                        tic = time.perf_counter()
                        df1dx1[row, column] += self.diffs['f1_X1'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df2dx1[row, column] += self.diffs['f2_X1'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)

                        df1dx2[row, column] += self.diffs['f1_X2'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df2dx2[row, column] += self.diffs['f2_X2'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)

                        toc = time.perf_counter()
                        lam_eval_time += toc - tic
        R1 = hstack([df1dx1.tocsr(), df1dx2.tocsr()])
        R2 = hstack([df2dx1.tocsr(), df2dx2.tocsr()])

        J = vstack([R1, R2])
        # print('lamdify_diff_time=', lamdify_diffs_time)
        print('lam_eval_time=', lam_eval_time)
        return J.tocsr()

    def Sparse_Jacobian_deprecated(self, unp1, vnp1, pnp1=None):
        """ This function is deprecated"""
        # numpy_unp1,numpy_vnp1 = self.__augment_fields_size(unp1,vnp1)
        numpy_unp1,numpy_vnp1 = self.augment_fields_size_NSE(unp1,vnp1)
        nx = self.prob_description.nx
        ny = self.prob_description.ny

        size = (nx) * (ny)
        # dfdx = csr_matrix(np.zeros([size,size]))
        df1dx1 = lil_matrix(np.zeros([size, size]))
        df2dx1 = lil_matrix(np.zeros([size, size]))
        df3dx1 = lil_matrix(np.zeros([size, size]))
        df1dx2 = lil_matrix(np.zeros([size, size]))
        df2dx2 = lil_matrix(np.zeros([size, size]))
        df3dx2 = lil_matrix(np.zeros([size, size]))
        df1dx3 = lil_matrix(np.zeros([size, size]))
        df2dx3 = lil_matrix(np.zeros([size, size]))
        df3dx3 = lil_matrix(np.zeros([size, size]))

        lam_eval_time = 0
        idxer = self.Indexer(nx, ny)
        for index_j in range(1, ny + 1):
            for index_i in range(1, nx + 1):
                row = idxer.linear_indexer_row(index_j, index_i)  # row in the jacobian matrix
                local_idxer_column = idxer.linear_indexer_column(index_j,index_i)  # return a function that takes a stencil point and return column in Jacobian matrix
                # for J, I in [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]:
                for J, I in [(-1, 0),(-1, 1), (0, -1), (0, 0), (0, 1),(0, 2),(1, -1),(1, 0),(1, 1),(2, 0)]:
                # for J in range(-1, 1 + 2): # plus two because the stencile is from -1 to +2
                #     for I in range(-1, 1 + 2): # plus two because the stencile is from -1 to +2
                        column = local_idxer_column(J, I)

                        tic = time.perf_counter()
                        df1dx1[row, column] += self.diffs['f1_X1'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df2dx1[row, column] += self.diffs['f2_X1'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df3dx1[row, column] += self.diffs['f3_X1'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)

                        df1dx2[row, column] += self.diffs['f1_X2'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df2dx2[row, column] += self.diffs['f2_X2'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df3dx2[row, column] += self.diffs['f3_X2'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)

                        df1dx3[row, column] += self.diffs['f1_X3'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df2dx3[row, column] += self.diffs['f2_X3'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        df3dx3[row, column] += self.diffs['f3_X3'][(J, I)](numpy_unp1, numpy_vnp1, index_j,
                                                                                       index_i)
                        toc = time.perf_counter()
                        lam_eval_time += toc - tic
        R1 = hstack([df1dx1.tocsr(), df1dx2.tocsr(), df1dx3.tocsr()])
        R2 = hstack([df2dx1.tocsr(), df2dx2.tocsr(), df2dx3.tocsr()])
        R3 = hstack([df3dx1.tocsr(), df3dx2.tocsr(), df3dx3.tocsr()])

        J = vstack([R1, R2, R3])
        # print('lamdify_diff_time=', lamdify_diffs_time)
        print('lam_eval_time=', lam_eval_time)
        return J.tocsr()

    def Sparse_Jacobian_scalar(self,phi_np1):
        """ This function is deprecated"""
        nx = self.prob_description.nx
        ny = self.prob_description.ny
        size = (nx) * (ny)
        dfdx = lil_matrix(np.zeros([size, size]))
        lam_eval_time = 0
        idxer = self.Indexer(nx, ny)                                                # smart indexer for periodic domain
        for index_j in range(1, ny +1):
            for index_i in range(1, nx +1):
                row = idxer.linear_indexer_row(index_j, index_i)                    # row in the jacobian matrix
                local_idxer_column = idxer.linear_indexer_column(index_j, index_i)  # return a function that takes a stencil point and return column in Jacobian matrix
                for J,I in [(-1,0),(0,-1),(0,0),(0,1),(1,0)]:
                    column = local_idxer_column(J, I)                               # column in the jacobian matrix
                    tic = time.perf_counter()
                    dfdx[row, column] = self.diffs['f_X'][(J, I)](phi_np1, index_j, index_i)
                    toc = time.perf_counter()
                    lam_eval_time += toc - tic

        J = dfdx.tocsr()
        print('lam_eval_time=', lam_eval_time)
        return J
