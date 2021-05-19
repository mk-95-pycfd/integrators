import sympy as sp
from sympy import IndexedBase,Idx


class SymResidualFunc:
    def __init__(self,problem_description):
        self.dt = problem_description.dt
        self.nu = problem_description.μ
        self.dx = problem_description.dx
        self.dy = problem_description.dy

        self.unp1 = IndexedBase('unp1')
        self.vnp1 = IndexedBase('vnp1')
        self.un = IndexedBase('un')
        self.vn = IndexedBase('vn')
        self.pnp1 = IndexedBase('pnp1')
        self.i, self.j = sp.symbols('i j', cls=Idx)

        self.__create_rhs_expressions()

    def __create_rhs_expressions(self):
        unp1 = self.unp1
        vnp1 = self.vnp1
        un = self.un
        vn = self.vn
        pnp1 = self.pnp1
        i, j = self.i, self.j

        dt = self.dt
        nu = self.nu
        dx = self.dx
        dy = self.dy

        # x convection
        self.__duudx = ((unp1[j, i + 1] + unp1[j, i]) ** 2 - (unp1[j, i] + unp1[j, i - 1]) ** 2) / 4 / dx
        self.__duvdy = ((unp1[j + 1, i] + unp1[j, i]) * (vnp1[j + 1, i - 1] + vnp1[j + 1, i]) - (
                    unp1[j - 1, i] + unp1[j, i]) * (vnp1[j, i - 1] + vnp1[j, i])) / 4 / dy
        # x diffusion
        self.__d2udx2 = nu * (unp1[j, i + 1] - 2 * unp1[j, i] + unp1[j, i - 1]) / dx / dx
        self.__d2udy2 = nu * (unp1[j + 1, i] - 2 * unp1[j, i] + unp1[j - 1, i]) / dy / dy
        # x pressure gradient
        self.__dpdx = (pnp1[j, i] - pnp1[j, i - 1]) / dx

        # y convection
        self.__duvdx = ((unp1[j, i + 1] + unp1[j - 1, i + 1]) * (vnp1[j, i + 1] + vnp1[j, i]) - (
                    unp1[j, i] + unp1[j - 1, i]) * (vnp1[j, i] + vnp1[j, i - 1])) / 4 / dx
        self.__dvvdy = ((vnp1[j + 1, i] + vnp1[j, i]) ** 2 - (vnp1[j, i] + vnp1[j - 1, i]) ** 2) / 4 / dy
        # y diffusion
        self.__d2vdx2 = nu * (vnp1[j, i + 1] - 2 * vnp1[j, i] + vnp1[j, i - 1]) / dx / dx
        self.__d2vdy2 = nu * (vnp1[j + 1, i] - 2 * vnp1[j, i] + vnp1[j - 1, i]) / dy / dy
        # y pressure gradient
        self.__dpdy = (pnp1[j, i] - pnp1[j - 1, i]) / dy

        # RHS pressure Poisson equation
        self.__div_un = ((un[j, i + 1] - un[j, i]) / dx + (vn[j + 1, i] - vn[j, i]) / dy)

        self.__div_convection = ((self.__duudx.subs({i: i + 1}) + self.__duvdy.subs({i: i + 1})) - (self.__duudx + self.__duvdy)) / dx \
                         + ((self.__duvdx.subs({j: j + 1}) + self.__dvvdy.subs({j: j + 1})) - (self.__duvdx + self.__dvvdy)) / dy

        self.__div_diffusion = ((self.__d2udx2.subs({i: i + 1}) + self.__d2udy2.subs({i: i + 1})) - (self.__d2udx2 + self.__d2udy2)) / dx \
                        + ((self.__d2vdx2.subs({j: j + 1}) + self.__d2vdy2.subs({j: j + 1})) - (self.__d2vdx2 + self.__d2vdy2)) / dy

        self.__laplacian_p = (pnp1[j, i + 1] - 2 * pnp1[j, i] + pnp1[j, i - 1]) / dx / dx + (
                    pnp1[j + 1, i] - 2 * pnp1[j, i] + pnp1[j - 1, i]) / dy / dy

    def full_rhs(self):
        rhs_uh = -(self.__duudx+self.__duvdy) + (self.__d2udx2+self.__d2udy2)
        rhs_u = rhs_uh - self.__dpdx

        rhs_vh = -(self.__duvdx + self.__dvvdy) + (self.__d2vdx2+self.__d2vdy2)
        rhs_v = rhs_vh - self.__dpdy

        rhs_p = - self.__div_convection + self.__div_diffusion - self.__laplacian_p

        return rhs_u, rhs_v, rhs_p

    def Stokes_rhs(self):
        rhs_uh =  (self.__d2udx2 + self.__d2udy2)
        rhs_u = rhs_uh - self.__dpdx

        rhs_vh =  (self.__d2vdx2 + self.__d2vdy2)
        rhs_v = rhs_vh - self.__dpdy

        rhs_p =  self.__div_diffusion - self.__laplacian_p

        return rhs_u, rhs_v, rhs_p

    def lhs(self):
        unp1 = self.unp1
        vnp1 = self.vnp1
        un = self.un
        vn = self.vn
        pnp1 = self.pnp1
        i, j = self.i, self.j

        dt = self.dt
        nu = self.nu
        dx = self.dx
        dy = self.dy

        lhs_u = (unp1[j, i] - un[j, i]) / dt

        lhs_v = (vnp1[j,i]-vn[j,i])/dt

        lhs_p = -((un[j,i+1]-un[j,i])/dx +(vn[j+1,i]-vn[j,i])/dy )/dt

        return lhs_u, lhs_v, lhs_p

    def vars(self):
        return self.unp1, self.vnp1, self.pnp1

    def all_vars(self):
        return self.unp1, self.vnp1,self.un, self.vn, self.pnp1

    def indices(self):
        return self.j,self.i



class SymResidualFunc_scalar:
    def __init__(self,problem_description):
        self.dt = problem_description.dt
        self.nu = problem_description.μ
        self.dx = problem_description.dx
        self.dy = problem_description.dy
        self.uf = problem_description.uf
        self.vf = problem_description.vf


        self.phi_np1 = IndexedBase('phi_np1')
        self.phi_n = IndexedBase('phi_n')
        self.i, self.j = sp.symbols('i j', cls=Idx)

    def rhs(self):
        phi_np1 = self.phi_np1
        i, j = self.i, self.j

        nu = self.nu
        dx = self.dx
        dy = self.dy
        uf = self.uf
        vf = self.vf
        conv = - uf * (phi_np1[j,i+1] - phi_np1[j,i-1])/2/dx - vf * (phi_np1[j+1,i] - phi_np1[j-1,i])/2/dy
        rhs_u = conv + nu*((phi_np1[j,i+1]-2*phi_np1[j,i]+phi_np1[j,i-1])/dx/dx +(phi_np1[j+1,i]-2*phi_np1[j,i]+phi_np1[j-1,i])/dy/dy)

        return rhs_u

    def lhs(self):
        phi_np1 = self.phi_np1
        phi_n = self.phi_n
        i, j = self.i, self.j

        dt = self.dt
        lhs_phi = (phi_np1[j, i] - phi_n[j, i]) / dt

        return lhs_phi

    def vars(self):
        return self.phi_np1

    def all_vars(self):
        return self.phi_np1,self.phi_n

    def indices(self):
        return self.j,self.i