from abc import ABC, abstractmethod

class PeriodicIndexer:

    _location=None

    def __init__(self,nx,ny):
        self.set_location(LLC(nx,ny))

    def set_location(self,location):
        self._location = location
        self._location.clocation = self

    def linear_indexer_column(self,j,i):
        return self._location.get_linear_index_column(j,i)

    def linear_indexer_row(self,j,i):
        return self._location.get_linear_index_row(j,i)

    def new_2d_indexer(self,j,i):
        return self._location.get_new_j_i_indices(j,i)

    def print_loc(self):
        print("↓---{}---↓\n----------------".format(self._location.location))


class IPeriodicIndexing(ABC):

    @property
    def clocation(self):
        return self._clocation

    @clocation.setter
    def clocation(self, new_location):
        self._clocation = new_location

    @property
    def location(self):
        return self.__class__.__name__

    @abstractmethod
    def get_linear_index_column(self,j,i):
        pass
    @abstractmethod
    def get_linear_index_row(self,j,i):
        pass

    @abstractmethod
    def get_new_j_i_indices(self,j,i):
        pass

    @abstractmethod
    def update_location(self,j,i):
        pass


class InnerDomain(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj,si):
            if si == 0 and sj >= 2 and j == self.ny - 1:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif sj == 0 and si >= 2 and i == self.nx - 1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj  - 1)
            else:
                return (i + si - 1 ) + (self.nx) * (j + sj - 1 )

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if si == 0 and sj >= 2 and j == self.ny - 1:
                return (j + sj - self.ny), (i + si)
            elif sj == 0 and si >= 2 and i == self.nx - 1:
                return (j + sj), (i + si - self.nx)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices


    def get_linear_index_row(self,j,i):
            return (i - 1) + (self.nx) * (j - 1)

    def update_location(self,j,i):
        if i==self.nx-1:
            self.clocation.set_location(InnerRight(self.nx,self.ny))

class InnerLower(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj<=-1:
                return (i + si - 1) + (self.nx) * (j + sj + self.ny - 1)
            elif sj==0 and si >=2 and i == self.nx -1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj<=-1:
                return (j + sj + self.ny), (i + si)
            elif sj==0 and si >=2 and i == self.nx -1:
                return (j + sj), (i + si - self.nx)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self,j,i):
        if i==self.nx-1:
            # right lower corner
            self.clocation.set_location(LRC(self.nx, self.ny))

class LLC(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj <= -1 and si >=0:
                return (i + si - 1) + (self.nx) * (j + sj + self.ny - 1)
            elif sj <= -1 and si <=0:
                return (i + si + self.nx - 1) + (self.nx) * (j + sj + self.ny - 1)
            elif si <= -1 and sj >=0:
                return (i + si + self.nx - 1) + (self.nx) * (j + sj - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj <= -1 and si >= 0:
                return (j + sj + self.ny), (i + si)
            elif sj <= -1 and si <= 0:
                return (j + sj + self.ny), (i + si + self.nx)
            elif si <= -1 and sj >= 0:
                return (j + sj), (i + si + self.nx)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self,j,i):
        # Inner Lower
        self.clocation.set_location(InnerLower(self.nx, self.ny))

class LRC(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj <= -1 and si ==0:
                return (i + si - 1) + (self.nx) * (j + sj + self.ny - 1)
            elif si >= 1 and sj == 0:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si >= 1 and sj >= 1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            elif sj <= -1 and si >= 1: #? is this correct?
                return (i + si - self.nx - 1) + (self.nx) * (j + sj + self.ny - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj <= -1 and si ==0:
                return (j + sj + self.ny), (i + si)
            elif si >= 1 and sj == 0:
                return (j + sj), (i + si - self.nx)
            elif si >= 1 and sj >= 1:
                return (j + sj), (i + si - self.nx)
            elif sj <= -1 and si >= 1: #? is this correct?
                return (j + sj + self.ny), (i + si - self.nx)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self,j,i):
        # Inner left
        self.clocation.set_location(InnerLeft(self.nx, self.ny))

class InnerLeft(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if si <= -1 :
                return (i + si + self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si ==0 and sj >=2 and j ==self.ny-1:
                return (i + si  - 1) + (self.nx) * (j + sj -self.ny - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if si <= -1 :
                return (j + sj), (i + si + self.nx)
            elif si ==0 and sj >=2 and j ==self.ny-1:
                return (j + sj -self.ny), (i + si )
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self,j,i):
        # Inner Domain corner
        self.clocation.set_location(InnerDomain(self.nx, self.ny))


class InnerRight(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if si >= +1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si == 0 and sj >= 2 and j == self.ny - 1:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if si >= +1:
                return (j + sj), (i + si - self.nx)
            elif si == 0 and sj >= 2 and j == self.ny - 1:
                return (j + sj - self.ny), (i + si)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self, j, i):
        if j == self.ny-1:
            # Upper left corner
            self.clocation.set_location(ULC(self.nx, self.ny))
        else:
            # Inner left
            self.clocation.set_location(InnerLeft(self.nx, self.ny))

class ULC(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj >= +1 and si==0:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif sj >= +1 and si==-1:
                return (i + si+ self.nx - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif si <= -1:
                return (i + si + self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si <= -1 and sj >= +1: #? is this correct
                return (i + si + self.nx - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif si >= 1 and sj >= +1:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj >= +1 and si==0:
                return (j + sj - self.ny), (i + si)
            elif sj >= +1 and si==-1:
                return (j + sj - self.ny), (i + si+ self.nx)
            elif si <= -1:
                return (j + sj), (i + si + self.nx)
            elif si <= -1 and sj >= +1: #? is this correct
                return (j + sj - self.ny), (i + si + self.nx)
            elif si >= 1 and sj >= +1:
                return (j + sj - self.ny), (i + si)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self, j, i):
        # Inner upper
        self.clocation.set_location(InnerUpper(self.nx, self.ny))

class InnerUpper(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj >= 1:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif sj==0 and si >=2 and i == self.nx-1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj  - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j, i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj >= 1:
                return  (j + sj - self.ny), (i + si)
            elif sj==0 and si >=2 and i == self.nx-1:
                return (j + sj ), (i + si - self.nx)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self, j, i):
        if i==self.nx-1:
            # right upper corner
            self.clocation.set_location(RUC(self.nx, self.ny))

class RUC(IPeriodicIndexing):
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

    def get_linear_index_column(self,j,i):
        def linear_index_column(sj, si):
            if sj >= 1 and si==0:
                return (i + si - 1) + (self.nx) * (j + sj - self.ny - 1)
            elif si >= 1 and sj==0:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si >= 1 and sj >= 1:#? is this correct
                return (i + si - self.nx - 1) + (self.nx) * (j + sj- self.ny - 1)
            elif si >= 1 and sj <= -1:
                return (i + si - self.nx - 1) + (self.nx) * (j + sj - 1)
            elif si <= -1 and sj >= 1:
                return (i + si - 1) + (self.nx) * (j + sj- self.ny - 1)
            else:
                return (i + si - 1) + (self.nx) * (j + sj - 1)

        self.update_location(j,i)
        return linear_index_column

    def get_new_j_i_indices(self, j, i):
        def new_j_i_indices(sj,si):
            if sj >= 1 and si==0:
                return (j + sj - self.ny), (i + si)
            elif si >= 1 and sj==0:
                return (j + sj), (i + si - self.nx)
            elif si >= 1 and sj >= 1:#? is this correct
                return (j + sj- self.ny), (i + si - self.nx)
            elif si >= 1 and sj <= -1:
                return (j + sj), (i + si - self.nx)
            elif si <= -1 and sj >= 1:
                return (j + sj- self.ny), (i + si)
            else:
                return (j + sj), (i + si)
        return new_j_i_indices

    def get_linear_index_row(self,j,i):
        return (i - 1) + (self.nx) * (j - 1)

    def update_location(self, j, i):
        # last state
        pass


# import numpy as np
# import matplotlib.pyplot as plt
# nx = 4
# ny = 4
# data = np.zeros([(nx)*(ny),(nx)*(ny)])
# idxer = PeriodicIndexer(nx,ny)
# idxer.print_loc()
# for j in range(1,ny+1):
#     for i in range(1,nx+1):
#         print('<j,i> <{},{}>'.format(j, i))
#         row = idxer.linear_indexer_row(j,i)
#         new_j_i = idxer.new_2d_indexer(j, i)
#         local_idxer_column = idxer.linear_indexer_column(j,i)
#         # for sj,si in [(-1,0),(0,-1),(0,0),(0,1),(1,0)]: # scalar
#         for sj,si in [(-1, 0),(-1, 1), (0, -1), (0, 0), (0, 1),(0, 2),(1, -1),(1, 0),(1, 1),(2, 0)]: # NSE
#             column = local_idxer_column(sj,si)
#             new_j,new_i = new_j_i(sj,si)
#             print('({},{}) ---> ({},{}) ---> <row, column> <{},{}>'.format(j+sj,i+si,new_j,new_i,row,column))
#             data[row,column] = 1
#         idxer.print_loc()
# print(data)
# plt.spy(data)
# plt.show()
