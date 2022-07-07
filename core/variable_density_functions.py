import numpy as np
from scipy.optimize import newton
import scipy.sparse
import scipy.sparse.linalg
import pyamg
import time
from core.singleton_classes import RK4, RK3, RK2


class vardenfunc:
    def __init__(self, probDescription,bcs_type=None):
        self.probDescription = probDescription
        self.bcs_type = bcs_type

    def periodic_scalar(self, f):
        # set periodicity on the scalar
        f[-1, :] = f[1, :]
        f[0, :] = f[-2, :]
        f[:, -1] = f[:, 1]
        f[:, 0] = f[:, -2]

    def periodic_u(self, u):
        # set periodicity in x
        u[:, -1] = u[:, 1]
        u[:, 0] = u[:, -2]
        # set periodicity in y
        u[-1, :] = u[1, :]
        u[0, :] = u[-2, :]

    def periodic_v(self, v):
        # set periodicity in y
        v[-1, :] = v[1, :]
        v[0, :] = v[-2, :]
        # set periodicity in x
        v[:, -1] = v[:, 1]
        v[:, 0] = v[:, -2]

    def View(self,field,direction):
        if direction == "P":
            return field[1:-1,1:-1]
        elif direction =="E":
            return field[1:-1,2:]
        elif direction == "W":
            return field[1:-1,:-2]
        elif direction == "N":
            return field[2:,1:-1]
        elif direction == "S":
            return field[:-2,1:-1]
        elif direction == "NE":
            return field[2:,2:]
        elif direction == "NW":
            return field[2:,:-2]
        elif direction == "SE":
            return field[:-2,2:]
        elif direction == "SW":
            return field[:-2,:-2]

    def XfluxView(self,field,direction):
        if direction == "e":
            return self.View(field, "E")
        elif direction =="w":
            return self.View(field, "P")

    def YfluxView(self,field,direction):
        if direction == "n":
            return self.View(field, "N")
        elif direction == "s":
            return self.View(field, "P")

    def interpSvol(self,phi,direction):
        if direction=="e":
            return 0.5 * (self.View(phi, "P") + self.View(phi, "E"))
        elif direction=="w":
            return 0.5 * (self.View(phi, "P") + self.View(phi, "W"))

        elif direction=="n":
            return 0.5 * (self.View(phi, "P") + self.View(phi, "N"))
        elif direction=="s":
            return 0.5 * (self.View(phi, "P") + self.View(phi, "S"))

    def interpYvol(self,phi,direction):
        if direction=="e": # interpolate the y velocity to the east cell corners
            return 0.5 * (self.View(phi, "P") + self.View(phi, "E"))
        elif direction=="w": # interpolate the y velocity to the west cell corners
            return 0.5 * (self.View(phi, "P") + self.View(phi, "W"))

    def interpXvol(self,phi,direction):
        if direction=="n": # interpolate the x velocity to the north cell corners ( it is the same as the east cell corners)
            return 0.5 * (self.View(phi, "P") + self.View(phi, "N"))
        elif direction=="s": # interpolate the x velocity to the south cell corners ( it is the same as the east cell corners)
            return 0.5 * (self.View(phi, "P") + self.View(phi, "S"))

    def div(self,u,v,phi):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        phie = self.interpSvol(phi, "e")
        phiw = self.interpSvol(phi, "w")
        phin = self.interpSvol(phi, "n")
        phis = self.interpSvol(phi, "s")

        ue = self.XfluxView(u, "e")  # uE
        uw = self.XfluxView(u, "w")  # uP
        un = self.YfluxView(v, "n")  # vN
        us = self.YfluxView(v, "s")  # vP

        div_u_phi = np.zeros_like(phi)
        div_u_phi[1:-1, 1:-1] = (ue * phie - uw * phiw) / dx  + (un * phin - us * phis) / dy
        return div_u_phi

    def div_vect(self,x_comp,y_comp):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        x_comp_e = self.XfluxView(x_comp, "e")  # E
        x_comp_w = self.XfluxView(x_comp, "w")  # P
        y_comp_n = self.YfluxView(y_comp, "n")  # N
        y_comp_s = self.YfluxView(y_comp, "s")  # P

        div_vect_phi = np.zeros_like(x_comp)
        div_vect_phi[1:-1,1:-1] = (x_comp_e - x_comp_w ) / dx + (y_comp_n - y_comp_s ) / dy
        return div_vect_phi

    def xConvection(self,u,v):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # ue = self.XfluxView(u, "e")
        # uw = self.XfluxView(u, "w")
        # un = self.interpXvol(u, "n")
        # us = self.interpXvol(u, "s")
        # vn =  self.YfluxView(v,"n")
        # vs =  self.YfluxView(v,"s")

        ue = 0.5 * (self.View(u,"E") + self.View(u,"P"))
        uw = 0.5 * (self.View(u,"W") + self.View(u,"P"))

        un = 0.5 * (self.View(u,"N") + self.View(u,"P"))
        us = 0.5 * (self.View(u,"S") + self.View(u,"P"))

        vn = 0.5 * (self.View(v,"NW") + self.View(v,"N"))
        vs = 0.5 * (self.View(v,"W") + self.View(v,"P"))

        convection = np.zeros_like(u)
        convection[1:-1, 1:-1] = - (ue * ue - uw * uw) / dx - (un * vn - us * vs) / dy
        return convection
        # return - (ue * ue - uw * uw) / dx - (un * vn - us * vs) / dy

    def yConvection(self,u,v):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # vn = self.YfluxView(v,"n")
        # vs = self.YfluxView(v,"s")
        # ve = self.interpYvol(v,"e")
        # vw = self.interpYvol(v,"w")
        # ue = self.XfluxView(u,"e")
        # uw = self.XfluxView(u,"w")

        ve = 0.5 * (self.View(v,"E") + self.View(v,"P"))
        vw = 0.5 * (self.View(v,"W") + self.View(v,"P"))

        ue = 0.5 * (self.View(u,"E") + self.View(u,"SE"))
        uw = 0.5 * (self.View(u,"P") + self.View(u,"S"))

        vn = 0.5 * (self.View(v,"N") + self.View(v,"P"))
        vs = 0.5 * (self.View(v,"S") + self.View(v,"P"))
        convection = np.zeros_like(u)
        convection[1:-1, 1:-1] = - (ue * ve - uw * vw) / dx - (vn * vn - vs * vs) / dy
        return convection
        # return - (ue * ve - uw * vw) / dx - (vn * vn - vs * vs) / dy

    def Newton_solver(self,guess,func, tol=1e-10,maxiter=100):
        shape = guess.shape
        guess_as_1d = guess.ravel()
        sol = newton(func,guess_as_1d,tol=tol,maxiter=maxiter)
        return sol.reshape(shape)

    def strain_xx(self,u):
        dx = self.probDescription.dx
        strain_xx_cc = np.zeros_like(u)
        strain_xx_cc[1:-1,1:-1] = - 2 * (self.View(u, "E") - self.View(u, "P"))/dx
        return strain_xx_cc

    def strain_yy(self,v):
        dy = self.probDescription.dy
        strain_yy_cc = np.zeros_like(v)
        strain_yy_cc[1:-1,1:-1] = - 2 * (self.View(v, "N") - self.View(v, "P"))/dy
        return strain_yy_cc

    def strain_xy(self,u,v):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        strain_xy_corner = np.zeros_like(u)
        strain_xy_corner[1:-1,1:-1] = - (self.View(v,"E") - self.View(v,"P"))/dx \
                                      - (self.View(u,"N") - self.View(u,"P"))/dy
        return strain_xy_corner

    def xMomPartialRHS(self,u,v,viscosity):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        xmom_partial = np.zeros_like(u)
        tempxx = viscosity * (self.strain_xx(u) + 2/3 * self.div_vect(u,v))
        tempyx = viscosity * self.strain_xy(u,v)
        # xmom_partial[1:-1,1:-1] = -(self.View(tempxx,"E") - self.View(tempxx,"P"))/dx \
        #                           - (self.View(tempyx,"N") - self.View(tempyx,"P"))/dy
        # add convection
        xmom_partial = self.xConvection(u,v) + viscosity * self.laplacian(u)
        return xmom_partial

    def yMomPartialRHS(self, u, v, viscosity):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        ymom_partial = np.zeros_like(v)
        tempyy = viscosity * (self.strain_yy(v) + 2/3 * self.div_vect(u,v))
        tempxy = viscosity * self.strain_xy(u, v)
        # ymom_partial[1:-1, 1:-1] = - (self.View(tempxy, "E") - self.View(tempxy, "P")) / dx \
        #                             -(self.View(tempyy, "N") - self.View(tempyy, "P")) / dy

        # add convection
        ymom_partial = self.yConvection(u,v) + viscosity * self.laplacian(v)
        return ymom_partial



    def A(self):
        # todo: need to to  account for density variation and different bcs
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        # build pressure coefficient matrix
        Ap = np.zeros([ny, nx])
        Ae = 1.0 / dx / dx * np.ones([ny, nx])
        As = 1.0 / dy / dy * np.ones([ny, nx])
        An = 1.0 / dy / dy * np.ones([ny, nx])
        Aw = 1.0 / dx / dx * np.ones([ny, nx])
        # # set left wall coefs
        Aw[:, 0] = 0.0
        # # set right wall coefs
        Ae[:, -1] = 0.0

        Awb = 1.0 / dx / dx * np.ones([ny, nx])
        Awb[:, 1:] = 0

        Asb = 1.0 / dx / dx * np.ones([ny, nx])
        Asb[1:, :] = 0

        Aeb = 1.0 / dx / dx * np.ones([ny, nx])
        Aeb[:, :-1] = 0

        Anb = 1.0 / dx / dx * np.ones([ny, nx])
        Anb[:-1, :] = 0

        Ap = -(Aw + Ae + An + As + Awb + Aeb)

        n = nx * ny
        d0 = Ap.reshape(n)
        # print(d0)
        de = Ae.reshape(n)[:-1]
        # print(de)
        dw = Aw.reshape(n)[1:]
        # print(dw)
        ds = As.reshape(n)[nx:]
        # print(ds)
        dn = An.reshape(n)[:-nx]
        # print(dn)
        dwb = Awb.reshape(n)[:-nx + 1]
        # print(dwb)
        dsb = Asb.reshape(n)[:nx]
        # print(dsb)
        deb = Aeb.reshape(n)[nx - 1:]
        # print(deb)
        dnb = Anb.reshape(n)[-nx:]
        # print(dnb)
        A1 = scipy.sparse.diags([d0, de, dw, dn, ds, dwb, dsb, deb, dnb],
                                [0, 1, -1, nx, -nx, nx - 1, nx * (ny - 1), -nx + 1, -nx * (ny - 1)], format='csr')
        return A1

    def Psolve(self,uh,vh,ci,MatCoef,atol=1e-12):
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        divuhat = self.div_vect(uh, vh)
        prhs = divuhat[1:-1, 1:-1]/ dt / ci
        rhs = prhs.ravel()

        def solver(A, b):
            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=atol)
            return ptmp

        ptmp = solver(MatCoef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        return p

    def GradXScalar(self,field):
        dx = self.probDescription.dx
        gx = np.zeros_like(field)
        gx[1:-1,1:-1] = (self.View(field,"P") - self.View(field,"W"))/dx
        return gx

    def GradYScalar(self,field):
        dy = self.probDescription.dy
        gy = np.zeros_like(field)
        gy[1:-1, 1:-1] = (self.View(field,"P") - self.View(field,"S"))/dy
        return gy

    def ImQ(self, uh, vh, Coef, p0, ci=1, tol=None, atol=None):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = self.div_vect(uh, vh)

        prhs = 1.0 / dt/ci * divuhat[1:-1, 1:-1]

        rhs = prhs.ravel()

        def solver(A, b):
            num_iters = 0

            def callback(xk):
                nonlocal num_iters
                num_iters += 1

            ml = pyamg.ruge_stuben_solver(A)
            ptmp = ml.solve(b, tol=1e-12, callback=callback)
            return ptmp, num_iters

        ptmp, num_iters = solver(Coef, rhs)
        p = np.zeros([ny + 2, nx + 2])
        p[1:-1, 1:-1] = ptmp.reshape([ny, nx])

        # set periodicity on the pressure
        self.periodic_scalar(p)

        # time advance
        unp1[1:-1,1:-1] = uh[1:-1,1:-1] - ci * dt * (self.View(p,"P") - self.View(p,"W"))/dx
        vnp1[1:-1,1:-1] = vh[1:-1,1:-1] - ci * dt * (self.View(p,"P") - self.View(p,"S")) / dy
        # unp1[1:-1, 1:] = uh[1:-1, 1:] - ci*dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        # vnp1[1:, 1:-1] = vh[1:, 1:-1] - ci*dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters

    def laplacian(self,phi):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        lap = np.zeros_like(phi)

        lap[1:-1, 1:-1] = ((self.View(phi, "E") - 2.0 * self.View(phi, "P") + self.View(phi, "W")) / dx / dx + (
                self.View(phi, "N") - 2.0 * self.View(phi, "P") + self.View(phi, "S")) / dy / dy)
        return lap

    def scalar_rhs(self,diff_coef,u,v,phi):
        return - self.div(u,v,phi) + diff_coef * self.laplacian(phi)