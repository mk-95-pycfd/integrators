import numpy as np
from scipy.optimize import newton
import scipy.sparse
import scipy.sparse.linalg
import pyamg

class SpatialOperators:
    probDescription = None

    @staticmethod
    def set_problemDescription(value):
        SpatialOperators.probDescription = value

    class FieldSlice:
        P = np.s_[1:-1,1:-1]
        E = np.s_[1:-1,2:]
        W = np.s_[1:-1,:-2]
        N = np.s_[2:,1:-1]
        S = np.s_[:-2,1:-1]
        NE = np.s_[2:, 2:]
        NW = np.s_[2:, :-2]
        SE = np.s_[:-2, 2:]
        SW = np.s_[:-2, :-2]

    @staticmethod
    def View(field, direction):
        if direction == "P":
            return field[SpatialOperators.FieldSlice.P]
        elif direction == "E":
            return field[SpatialOperators.FieldSlice.E]
        elif direction == "W":
            return field[SpatialOperators.FieldSlice.W]
        elif direction == "N":
            return field[SpatialOperators.FieldSlice.N]
        elif direction == "S":
            return field[SpatialOperators.FieldSlice.S]
        elif direction == "NE":
            return field[SpatialOperators.FieldSlice.NE]
        elif direction == "NW":
            return field[SpatialOperators.FieldSlice.NW]
        elif direction == "SE":
            return field[SpatialOperators.FieldSlice.SE]
        elif direction == "SW":
            return field[SpatialOperators.FieldSlice.SW]

    @staticmethod
    def Interpolate(field,face):
        face_field = np.zeros_like(field)

        if face == "e":
            face_field[SpatialOperators.FieldSlice.P] = 0.5 * (field[SpatialOperators.FieldSlice.P] + field[SpatialOperators.FieldSlice.E])
        elif face == "w":
            face_field[SpatialOperators.FieldSlice.P] = 0.5 * (field[SpatialOperators.FieldSlice.P] + field[SpatialOperators.FieldSlice.W])
        elif face == "n":
            face_field[SpatialOperators.FieldSlice.P] = 0.5 * (field[SpatialOperators.FieldSlice.P] + field[SpatialOperators.FieldSlice.N])
        elif face == "s":
            face_field[SpatialOperators.FieldSlice.P] = 0.5 * (field[SpatialOperators.FieldSlice.P] + field[SpatialOperators.FieldSlice.S])

        return face_field

    class Calculus:
        @staticmethod
        def div_vect(x_comp, y_comp):
            dx = SpatialOperators.probDescription.dx
            dy = SpatialOperators.probDescription.dy

            x_comp_e = SpatialOperators.View(x_comp, "E")
            x_comp_w = SpatialOperators.View(x_comp, "P")
            y_comp_n = SpatialOperators.View(y_comp, "N")
            y_comp_s = SpatialOperators.View(y_comp, "P")

            div_vect_phi = np.zeros_like(x_comp)
            div_vect_phi[SpatialOperators.FieldSlice.P] = (x_comp_e - x_comp_w) / dx + (y_comp_n - y_comp_s) / dy
            return div_vect_phi

        @staticmethod
        def flux(velocity_vect,Scalar, face):
            velocity_x, velocity_y = velocity_vect
            face_flux = np.zeros_like(Scalar)

            if face == "e":
                velocity_x_east = SpatialOperators.View(velocity_x, "E")
                scalar_east = SpatialOperators.View(SpatialOperators.Interpolate(Scalar,"e"),"P")
                face_flux[SpatialOperators.FieldSlice.P] = velocity_x_east * scalar_east
            elif face == "w":
                velocity_x_west = SpatialOperators.View(velocity_x, "W")
                scalar_west = SpatialOperators.View(SpatialOperators.Interpolate(Scalar, "w"),"P")
                face_flux[SpatialOperators.FieldSlice.P] = velocity_x_west * scalar_west
            elif face == "n":
                velocity_y_north = SpatialOperators.View(velocity_y, "N")
                scalar_north = SpatialOperators.View(SpatialOperators.Interpolate(Scalar, "n"),"P")
                face_flux[SpatialOperators.FieldSlice.P] = velocity_y_north * scalar_north
            elif face == "s":
                velocity_y_south = SpatialOperators.View(velocity_y, "S")
                scalar_south = SpatialOperators.View(SpatialOperators.Interpolate(Scalar, "s"),"P")
                face_flux[SpatialOperators.FieldSlice.P] = velocity_y_south * scalar_south

            return face_flux

        @staticmethod
        def div_flux(u, v, phi):
            dx = SpatialOperators.probDescription.dx
            dy = SpatialOperators.probDescription.dy

            flux_east = SpatialOperators.View(SpatialOperators.Calculus.flux((u,v),phi,"e"),"P")
            flux_west = SpatialOperators.View(SpatialOperators.Calculus.flux((u,v),phi,"w"),"P")
            flux_north = SpatialOperators.View(SpatialOperators.Calculus.flux((u,v),phi,"n"),"P")
            flux_south = SpatialOperators.View(SpatialOperators.Calculus.flux((u,v),phi,"s"),"P")

            div_u_phi = np.zeros_like(phi)
            div_u_phi[SpatialOperators.FieldSlice.P] = (flux_east - flux_west) / dx + (flux_north - flux_south) / dy
            return div_u_phi

        @staticmethod
        def GradXScalar(field):
            dx = SpatialOperators.probDescription.dx
            gx = np.zeros_like(field)
            gx[SpatialOperators.FieldSlice.P] = (SpatialOperators.View(field, "P") - SpatialOperators.View(field, "W")) / dx
            return gx

        @staticmethod
        def GradYScalar(field):
            dy = SpatialOperators.probDescription.dy
            gy = np.zeros_like(field)
            gy[SpatialOperators.FieldSlice.P] = (SpatialOperators.View(field, "P") - SpatialOperators.View(field, "S")) / dy
            return gy

class BoundaryConditions:

    @staticmethod
    def periodic_x(f):
        # set periodicity in x
        f[:, -1] = f[:, 1]
        f[:, 0] = f[:, -2]

    @staticmethod
    def periodic_y(f):
        # set periodicity in y
        f[-1, :] = f[1, :]
        f[0, :] = f[-2, :]

    @staticmethod
    def periodic(f):
        # set periodicity in x
        BoundaryConditions.periodic_x(f)
        # set periodicity in y
        BoundaryConditions.periodic_y(f)

class vardenfunc:
    def __init__(self, probDescription,bcs_type=None):
        self.probDescription = probDescription
        self.bcs_type = bcs_type

    def timeStepSelector(self,u,v,viscosity,density,inner_scale=1,outer_scale=1):
        #compute the local cell Reynolds numbers
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        rho = SpatialOperators.View(density,"P")
        mu = SpatialOperators.View(viscosity,"P")
        ucc = 0.5 * np.abs(SpatialOperators.View(u,"P") + SpatialOperators.View(u,"E"))
        vcc = 0.5 * np.abs(SpatialOperators.View(v,"P") + SpatialOperators.View(v,"N"))
        Rex = rho*ucc*dx/mu
        Rey = rho*vcc*dy/mu
        u_dx = ucc/dx
        v_dy = vcc/dy

        # FE
        sum_outer = (u_dx * Rex + v_dy * Rey) / 2
        sum_inner = 2 * (mu/rho/dx/dx + mu/rho/dy/dy)
        dt_inner_local = inner_scale/sum_inner
        dt_outer_local = outer_scale/sum_outer
        dt_local = np.minimum(dt_inner_local,dt_outer_local)
        stable_dt = np.min(dt_local)
        print("Crx={}, Cry={}".format(np.max(ucc*stable_dt/dx),np.max(vcc*stable_dt/dy)))
        self.probDescription.set_dt(stable_dt)

    def xConvection(self,u,v,density):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        density_e = SpatialOperators.View(density,"P") # represents the east value
        density_w = SpatialOperators.View(density,"W")
        density_n = 0.25 * ( SpatialOperators.View(density,"P") + SpatialOperators.View(density,"N")
                             + SpatialOperators.View(density,"NW") + SpatialOperators.View(density,"W") )
        density_s = 0.25 * ( SpatialOperators.View(density,"P") + SpatialOperators.View(density,"S")
                             + SpatialOperators.View(density,"SW") + SpatialOperators.View(density,"W") )

        ue = 0.5 * (SpatialOperators.View(u,"E") + SpatialOperators.View(u,"P"))
        uw = 0.5 * (SpatialOperators.View(u,"W") + SpatialOperators.View(u,"P"))

        un = 0.5 * (SpatialOperators.View(u,"N") + SpatialOperators.View(u,"P"))
        us = 0.5 * (SpatialOperators.View(u,"S") + SpatialOperators.View(u,"P"))

        vn = 0.5 * (SpatialOperators.View(v,"NW") + SpatialOperators.View(v,"N"))
        vs = 0.5 * (SpatialOperators.View(v,"W") + SpatialOperators.View(v,"P"))

        convection = np.zeros_like(u)
        convection[1:-1, 1:-1] = - (ue * (density_e * ue) - uw * (density_w * uw)) / dx - (vn * (un * density_n)   - vs * (density_s * us) ) / dy
        return convection

    def yConvection(self,u,v,density):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        density_e =  0.25 * (SpatialOperators.View(density, "P") + SpatialOperators.View(density, "E")
                            + SpatialOperators.View(density, "S") + SpatialOperators.View(density, "SE"))

        density_w = 0.25 * (SpatialOperators.View(density, "P") + SpatialOperators.View(density, "W")
                            + SpatialOperators.View(density, "SW") + SpatialOperators.View(density, "S"))

        density_n = SpatialOperators.View(density,"P")
        density_s = SpatialOperators.View(density,"S")

        ve = 0.5 * (SpatialOperators.View(v,"E") + SpatialOperators.View(v,"P"))
        vw = 0.5 * (SpatialOperators.View(v,"W") + SpatialOperators.View(v,"P"))

        ue = 0.5 * (SpatialOperators.View(u,"E") + SpatialOperators.View(u,"SE"))
        uw = 0.5 * (SpatialOperators.View(u,"P") + SpatialOperators.View(u,"S"))

        vn = 0.5 * (SpatialOperators.View(v,"N") + SpatialOperators.View(v,"P"))
        vs = 0.5 * (SpatialOperators.View(v,"S") + SpatialOperators.View(v,"P"))
        convection = np.zeros_like(u)
        convection[1:-1, 1:-1] = - (ue * (density_e * ve) - uw * (density_w * vw)) / dx - (vn * (density_n * vn) - vs * (density_s * vs)) / dy
        return convection

    def Newton_solver(self,guess,func, tol=1e-10,maxiter=100):
        shape = guess.shape
        guess_as_1d = guess.ravel()
        sol = newton(func,guess_as_1d,tol=tol,maxiter=maxiter)
        return sol.reshape(shape)

    ## to avoid applying bcs on the strain we write the momrhs in terms of the velocity
    #  by assuming constant viscosity across the domain
    # def strain_xx(self,u,v,viscosity):
    #     dx = self.probDescription.dx
    #     strain_xx_cc = np.zeros_like(u)
    #     strain_xx_cc[SpatialOperators.FieldSlice.P] = viscosity[SpatialOperators.FieldSlice.P]*(- 2 * (SpatialOperators.View(u, "E") - SpatialOperators.View(u, "P"))/dx \
    #                               + 2/3 * SpatialOperators.View(SpatialOperators.Calculus.div_vect(u,v),"P"))
    #     return strain_xx_cc
    #
    # def strain_yy(self,u,v,viscosity):
    #     dy = self.probDescription.dy
    #     strain_yy_cc = np.zeros_like(v)
    #     strain_yy_cc[SpatialOperators.FieldSlice.P] =  viscosity[SpatialOperators.FieldSlice.P]*(- 2 * (SpatialOperators.View(v, "N") - SpatialOperators.View(v, "P"))/dy
    #                                                                                             + 2/3 * SpatialOperators.View(SpatialOperators.Calculus.div_vect(u,v),"P"))
    #     return strain_yy_cc
    #
    # def strain_xy(self,u,v,viscosity):
    #     dx = self.probDescription.dx
    #     dy = self.probDescription.dy
    #
    #     dvdx = (SpatialOperators.View(v,"E") + SpatialOperators.View(v,"NE") - SpatialOperators.View(v,"W") - SpatialOperators.View(v,"NW"))/dx/4
    #     dudy =  (SpatialOperators.View(u,"N") + SpatialOperators.View(u,"NE") - SpatialOperators.View(u,"S") - SpatialOperators.View(u,"SE"))/dy/4
    #     strain_xy_cc = np.zeros_like(u)
    #     strain_xy_cc[SpatialOperators.FieldSlice.P] = - viscosity[SpatialOperators.FieldSlice.P]*(dvdx + dudy)
    #     return strain_xy_cc

    def xMomPartialRHS(self,u,v,viscosity,density):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dudxx = np.zeros_like(u)
        dvdydx = np.zeros_like(u)
        dudxx[SpatialOperators.FieldSlice.P] =  (SpatialOperators.View(u, "E") - 2.0 * SpatialOperators.View(u, "P")
                                                 + SpatialOperators.View(u, "W")) / dx / dx
        dvdydx[SpatialOperators.FieldSlice.P] = (SpatialOperators.View(v, "N") - SpatialOperators.View(v, "P")
                                                 + SpatialOperators.View(v, "W") - SpatialOperators.View(v, "NW")) / dx / dy

        xmom_partial = self.xConvection(u,v,density) + viscosity *(self.laplacian(u) + 1/3 *(dudxx + dvdydx ))
        return xmom_partial

    def yMomPartialRHS(self, u, v, viscosity,density):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dvdyy = np.zeros_like(u)
        dudydx = np.zeros_like(u)

        dvdyy[SpatialOperators.FieldSlice.P] = (SpatialOperators.View(v, "N") - 2.0 * SpatialOperators.View(v, "P")
                                                + SpatialOperators.View(v,"S")) / dy / dy
        dudydx[SpatialOperators.FieldSlice.P] = (SpatialOperators.View(u, "E") - SpatialOperators.View(u, "P")
                                                 + SpatialOperators.View(u,"S") - SpatialOperators.View(u, "SE")) / dx / dy

        # add convection
        ymom_partial = self.yConvection(u,v,density) + viscosity * (self.laplacian(v) + 1/3 *(dudydx + dvdyy ))
        return ymom_partial



    def A(self,density):
        # todo: need to to  account for density variation and different bcs
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        nx = self.probDescription.nx
        ny = self.probDescription.ny

        density_E = SpatialOperators.View(density,"E")
        density_W = SpatialOperators.View(density,"W")
        density_N = SpatialOperators.View(density,"N")
        density_S = SpatialOperators.View(density,"S")
        density_P = SpatialOperators.View(density,"P")
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

        Ae = Ae/ density_E
        As = As/ density_S
        An = An/ density_N
        Aw = Aw/ density_W
        Ap = Ap/ density_P

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

        divuhat = SpatialOperators.Calculus.div_vect(uh, vh)
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

    def ImQ(self, uh, vh, Coef, p0, ci=1, tol=None, atol=None):
        dx = self.probDescription.dx
        dy = self.probDescription.dy
        dt = self.probDescription.dt
        nx = self.probDescription.nx
        ny = self.probDescription.ny
        unp1 = np.zeros_like(uh)
        vnp1 = np.zeros_like(vh)
        divuhat = SpatialOperators.Calculus.div_vect(uh, vh)

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
        unp1[1:-1,1:-1] = uh[1:-1,1:-1] - ci * dt * (SpatialOperators.View(p,"P") - SpatialOperators.View(p,"W"))/dx
        vnp1[1:-1,1:-1] = vh[1:-1,1:-1] - ci * dt * (SpatialOperators.View(p,"P") - SpatialOperators.View(p,"S")) / dy
        # unp1[1:-1, 1:] = uh[1:-1, 1:] - ci*dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
        # vnp1[1:, 1:-1] = vh[1:, 1:-1] - ci*dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dy

        self.periodic_u(unp1)
        self.periodic_v(vnp1)

        return unp1, vnp1, p, num_iters

    def laplacian(self,phi):
        dx = self.probDescription.dx
        dy = self.probDescription.dy

        lap = np.zeros_like(phi)

        lap[1:-1, 1:-1] = ((SpatialOperators.View(phi, "E") - 2.0 * SpatialOperators.View(phi, "P") + SpatialOperators.View(phi, "W")) / dx / dx + (
                SpatialOperators.View(phi, "N") - 2.0 * SpatialOperators.View(phi, "P") + SpatialOperators.View(phi, "S")) / dy / dy)
        return lap

    def scalar_rhs(self,diff_coef,u,v,phi):
        return - SpatialOperators.Calculus.div_flux(u,v,phi) + diff_coef * self.laplacian(phi)

    def scalar_upwind_convection(self,u,v,phi):
        # weak form
        ue = 0.5 * (SpatialOperators.View(u, "E") + SpatialOperators.View(u, "P"))
        uw = 0.5 * (SpatialOperators.View(u, "W") + SpatialOperators.View(u, "P"))

        un = 0.5 * (SpatialOperators.View(u, "N") + SpatialOperators.View(u, "P"))
        us = 0.5 * (SpatialOperators.View(u, "S") + SpatialOperators.View(u, "P"))

        vn = 0.5 * (SpatialOperators.View(v, "NW") + SpatialOperators.View(v, "N"))
        vs = 0.5 * (SpatialOperators.View(v, "W") + SpatialOperators.View(v, "P"))

        xvel_plus = 0.5 * (u )
