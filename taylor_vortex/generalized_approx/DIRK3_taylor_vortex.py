import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc
from core.analytical_jacobian import Analytic_Jacobian
from core.sym_residual_functions import SymResidualFunc
from core.Jacobian_indexing import PeriodicIndexer


def DIRK3_taylor_vortex (steps=3, return_stability=False,name='Crouzeix34',guess=None, project=[1,1,1],alpha=0.99,theta=0.5, Tol=1e-8,post_projection=False):
    # problem description
    probDescription = sc.ProbDescription()
    dt = probDescription.dt
    DIRK3 = sc.DIRK3(name, theta)
    a11 = DIRK3.a11
    a21 = DIRK3.a21
    a22 = DIRK3.a22
    a31 = DIRK3.a31
    a32 = DIRK3.a32
    a33 = DIRK3.a33
    b1 = DIRK3.b1
    b2 = DIRK3.b2
    b3 = DIRK3.b3

    # symbolic residual functions constructions
    Res_funcs = SymResidualFunc(probDescription)
    lhs_u, lhs_v, lhs_p = Res_funcs.lhs()
    rhs_u, rhs_v, rhs_p = Res_funcs.Stokes_rhs()

    # unsteady residual stage 1
    # --------------------------
    sym_f1_s1 = dt*(lhs_u - a11*rhs_u)
    sym_f2_s1 = dt*(lhs_v - a11*rhs_v)
    sym_f3_s1 = dt*(lhs_p - a11*rhs_p)

    # stage 1 approx
    # ----------------
    sym_f1_s1_approx = dt * (lhs_u - a11 * rhs_u)
    sym_f2_s1_approx = dt * (lhs_v - a11 * rhs_v)

    # unsteady residual stage 2
    # --------------------------
    sym_f1_s2 = dt*(lhs_u - a22 * rhs_u)
    sym_f2_s2 = dt*(lhs_v - a22 * rhs_v)
    sym_f3_s2 = dt*(lhs_p - a22 * rhs_p)

    # stage 2 approx
    #----------------
    sym_f1_s2_approx = dt * (lhs_u - a22 * rhs_u)
    sym_f2_s2_approx = dt * (lhs_v - a22 * rhs_v)

    # unsteady residual stage 3
    # --------------------------
    sym_f1_s3 = dt * (lhs_u - a33 * rhs_u)
    sym_f2_s3 = dt * (lhs_v - a33 * rhs_v)
    sym_f3_s3 = dt * (lhs_p - a33 * rhs_p)

    # stage 3 approx
    # ----------------
    sym_f1_s3_approx = dt * (lhs_u - a33 * rhs_u)
    sym_f2_s3_approx = dt * (lhs_v - a33 * rhs_v)

    x1, x2, x3 = Res_funcs.vars()

    j, i = Res_funcs.indices()

    stencil_pts = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
    stencil_pts_approx = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    # Jacobian Builder for every stage
    Jacobian_builder_s1 = Analytic_Jacobian([sym_f1_s1, sym_f2_s1, sym_f3_s1], [x1, x2, x3], [j, i], [-1, 1], stencil_pts,
                                         probDescription, PeriodicIndexer)

    Jacobian_builder_s1_approx = Analytic_Jacobian([sym_f1_s1_approx, sym_f2_s1_approx], [x1, x2], [j, i], [-1, 1],
                                            stencil_pts_approx,
                                            probDescription, PeriodicIndexer)

    Jacobian_builder_s2 = Analytic_Jacobian([sym_f1_s2, sym_f2_s2, sym_f3_s2], [x1, x2, x3], [j, i], [-1, 1], stencil_pts,
                                            probDescription, PeriodicIndexer)

    Jacobian_builder_s2_approx = Analytic_Jacobian([sym_f1_s2_approx, sym_f2_s2_approx], [x1, x2], [j, i], [-1, 1],
                                            stencil_pts_approx,
                                            probDescription, PeriodicIndexer)

    Jacobian_builder_s3 = Analytic_Jacobian([sym_f1_s3, sym_f2_s3, sym_f3_s3], [x1, x2, x3], [j, i], [-1, 1],
                                            stencil_pts,
                                            probDescription, PeriodicIndexer)

    Jacobian_builder_s3_approx = Analytic_Jacobian([sym_f1_s3_approx, sym_f2_s3_approx], [x1, x2], [j, i], [-1, 1],
                                                   stencil_pts_approx,
                                                   probDescription, PeriodicIndexer)

    f = func(probDescription, 'periodic')
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()
    # define exact solutions
    a = 2 * np.pi
    b = 2 * np.pi
    uf = 1
    vf = 1
    uexact = lambda a, b, x, y, t: uf - np.cos(a * (x - uf * t)) * np.sin(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)
    vexact = lambda a, b, x, y, t: vf + np.sin(a * (x - uf * t)) * np.cos(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)

    pexact = lambda x, y, t: (-8 * np.sin(np.pi * t) ** 4 * np.sin(np.pi * y) ** 4 - 2 * np.sin(
        np.pi * t) ** 4 - 2 * np.sin(np.pi * y) ** 4 - 5 * np.cos(
        2 * np.pi * t) / 2 + 5 * np.cos(4 * np.pi * t) / 8 - 5 * np.cos(2 * np.pi * y) / 2 + 5 * np.cos(
        4 * np.pi * y) / 8 - np.cos(
        np.pi * (2 * t - 4 * y)) / 4 + np.cos(np.pi * (2 * t - 2 * y)) + np.cos(np.pi * (2 * t + 2 * y)) - np.cos(
        np.pi * (2 * t + 4 * y)) / 4 - 3 * np.cos(np.pi * (4 * t - 4 * y)) / 16 - np.cos(
        np.pi * (4 * t - 2 * y)) / 4 - np.cos(
        np.pi * (4 * t + 2 * y)) / 4 + np.cos(np.pi * (4 * t + 4 * y)) / 16 + 27 / 8) * np.exp(
        -16 * np.pi ** 2 * μ * t) - np.exp(
        -16 * np.pi ** 2 * μ * t) * np.cos(np.pi * (-4 * t + 4 * x)) / 4

    #     # define some boiler plate
    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)

    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    u0[1:-1, 1:] = uexact(a, b, xu, yu, 0)  # initialize the interior of u0
    # same thing for the y-velocity component
    v0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    v0[1:, 1:-1] = vexact(a, b, xv, yv, 0)
    f.periodic_u(u0)
    f.periodic_v(v0)

    # initialize the pressure
    p0 = np.zeros([nx + 2, ny + 2]);  # include ghost cells
    p0[1:-1, 1:-1] = pexact(xcc, ycc, 0)
    f.periodic_scalar(p0)

    #declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1= np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol=[]
    usol.append(u0)

    vsol=[]
    vsol.append(v0)

    psol = []
    psol.append(p0)

    iterations = []

    Coef = f.A()

    is_stable =True
    stability_counter =0
    total_iteration =0
    # # u and v num cell centered
    ucc = 0.5*(u0[1:-1,2:] + u0[1:-1,1:-1])
    vcc = 0.5*(v0[2:,1:-1] + v0[1:-1,1:-1])

    uexc = uexact(a,b,xu,yu,t)
    vexc = vexact(a,b,xv,yv,t)
    # u and v exact cell centered
    uexc_cc = 0.5*(uexc[:,:-1] + uexc[:,1:])
    vexc_cc = 0.5*(vexc[:-1,:] + vexc[1:,:])

    # compute of kinetic energy
    ken_new = np.sum(ucc.ravel()**2 +vcc.ravel()**2)/2
    ken_exact = np.sum(uexc_cc.ravel()**2 +vexc_cc.ravel()**2)/2
    ken_old = ken_new
    final_KE = nx*ny
    alpha = 0.999
    target_ke = ken_exact - alpha*(ken_exact-final_KE)
    print('time = ',t)
    print('ken_new = ',ken_new)
    print('ken_exc = ',ken_exact)

    info_resid = {}

    # compute the Jacobian based on the Stokes equations since it doesn't change per timestep.
    Jacobian_s1 = Jacobian_builder_s1.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])
    Jacobian_s2 = Jacobian_builder_s2.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])
    Jacobian_s3 = Jacobian_builder_s3.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])

    Jacobian_s1_approx = Jacobian_builder_s1_approx.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])
    Jacobian_s2_approx = Jacobian_builder_s2_approx.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])
    Jacobian_s3_approx = Jacobian_builder_s3_approx.Sparse_Jacobian(usol[-1], vsol[-1], psol[-1])


    while count < tend:
        print('timestep:{}'.format(count+1))
        time_start = time.time()
        un = usol[-1].copy()
        vn = vsol[-1].copy()
        pn = psol[-1].copy()
        pnm1 = np.zeros_like(un)
        pnm2 = np.zeros_like(un)
        if count > 2:
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            pnm2 = psol[-3].copy()
            d1,d2,d3 = project

        elif count <= 2:  # compute pressures for 2 time steps # change to 2 if high order pressure is needed
            d1 = 1
            d2 = 1
            d3 = 1

        info_stage_1 = None

        # stage 1:
        #---------
        if d1 == 1:
            # unsteady residual stage1
            # -------------------------
            f1_s1 = lambda uold, vold, pold: lambda u, v, p: dt * ((u - uold) / dt - a11 * (f.urhs(u, v) - f.Gpx(p)))
            f2_s1 = lambda uold, vold, pold: lambda u, v, p: dt * ((v - vold) / dt - a11 * (f.vrhs(u, v) - f.Gpy(p)))
            f3_s1 = lambda uold, vold, pold: lambda u, v, p: dt * (
                        -a11 * (-f.div(f.Gpx(p), f.Gpy(p)) + f.div(f.urhs(u, v), f.vrhs(u, v))) \
                +(f.div(u, v)- f.div(uold, vold)) / dt)
            old_s1 = [un, vn, pn]
            residuals_s1 = [f1_s1, f2_s1, f3_s1]
            guesses_s1 = [un, vn, pn]
            Tol_s1 = Tol

            sol1, iterations, error,info_1= f.Newton_solver(guesses_s1, old_s1, residuals_s1,
                                                 [f.periodic_u, f.periodic_v, f.periodic_scalar], Jacobian=Jacobian_s1,
                                                 Tol=Tol_s1, verbose=False)
            u1, v1, p1 = sol1
            info_stage_1 = info_1
        elif d1 == 0:
            # unsteady residual stage1
            # -------------------------
            f1_s1_approx = lambda uold, vold, pold: lambda u, v: dt * ((u - uold) / dt - a11 * f.urhs(u, v) + f.Gpx(pold))
            f2_s1_approx = lambda uold, vold, pold: lambda u, v: dt * ((v - vold) / dt - a11 * f.vrhs(u, v) + f.Gpy(pold))

            # second order approx
            #=====================
            if guess =="second":
                alpha = a11*(2*a11+3)/2
                beta = -a11*(a11+1/2)
                old_s1_approx = [un, vn, alpha*pn + beta * pnm1] # here the value of p is the pressure approxmation
            #======================
            # third order approx
            #======================
            elif guess =="third":
                alpha = a11 * (6 * a11**2 + 12*a11  + 11) / 6
                beta = -a11*(12*a11**2 + 18*a11 +7)/6
                gamma = a11*(a11**2+a11+1/3)
                old_s1_approx = [un, vn, alpha * pn + beta * pnm1 + gamma * pnm2]  # here the value of p is the pressure approxmation

            residuals_s1_approx = [f1_s1_approx, f2_s1_approx]
            guesses_s1_approx = [un, vn]
            Tol_s1 = Tol

            sol1, iterations, error, info_1_approx = f.Newton_solver(guesses_s1_approx, old_s1_approx, residuals_s1_approx,
                                                      [f.periodic_u, f.periodic_v, f.periodic_scalar],
                                                      Jacobian=Jacobian_s1_approx,
                                                      Tol=Tol_s1, verbose=False)
            u1, v1 = sol1
            p1 = old_s1_approx[-1]
            info_stage_1 = info_1_approx
        f.periodic_u(u1)
        f.periodic_v(v1)
        f.periodic_scalar(p1)

        print('     non-linear iterations at stage 1: ', iterations)
        print('     residual at stage 1: ', error)
        print('     divergence at stage 1: ', np.linalg.norm(f.div(u1, v1).ravel()))

        rhs_u1 = f.urhs(u1, v1) #- f.Gpx(p1)
        rhs_v1 = f.vrhs(u1, v1) #- f.Gpy(p1)

        info_stage_2=None

        # stage 2:
        #---------
        if d2 ==1:

            # unsteady residual stage2
            # -------------------------
            f1_s2 = lambda uold, vold, pold: lambda u, v, p: dt * (
                        (u - uold) / dt - a21 * rhs_u1 - a22 * (f.urhs(u, v) - f.Gpx(p)))
            f2_s2 = lambda uold, vold, pold: lambda u, v, p: dt * (
                        (v - vold) / dt - a21 * rhs_v1 - a22 * (f.vrhs(u, v) - f.Gpy(p)))
            f3_s2 = lambda uold, vold, pold: lambda u, v, p: dt * (
                        -a22 * (-f.div(f.Gpx(p), f.Gpy(p)) + f.div(f.urhs(u, v), f.vrhs(u, v))) \
                        # - a21 * f.div(rhs_u1, rhs_v1) - f.div(uold, vold) / dt)
                        - a21 * f.div(rhs_u1, rhs_v1) + (f.div(u, v)- f.div(uold, vold)) / dt)

            old_s2 = [un, vn, pn]
            residuals_s2 = [f1_s2, f2_s2, f3_s2]
            guesses_s2 = [un, vn, pn]
            Tol_s2 = Tol

            sol2, iterations, error, info_2 = f.Newton_solver(guesses_s2, old_s2, residuals_s2,
                                                      [f.periodic_u, f.periodic_v, f.periodic_scalar],
                                                      Jacobian=Jacobian_s2,
                                                      Tol=Tol_s2, verbose=False)
            u2, v2, p2 = sol2
            info_stage_2 = info_2
        elif d2==0:
            # unsteady residual stage2
            # -------------------------
            f1_s2_approx = lambda uold, vold, pold: lambda u, v: dt * (
                    (u - uold) / dt - a21 * rhs_u1 - a22 * f.urhs(u, v) + f.Gpx(pold))
            f2_s2_approx = lambda uold, vold, pold: lambda u, v: dt * (
                    (v - vold) / dt - a21 * rhs_v1 - a22 * f.vrhs(u, v) + f.Gpy(pold))

            # second order approx
            #=====================
            if guess == "second":
                c2 = a22+a21
                alpha =  a11*a21 + a21*a22 +a22**2 + 3*c2/2
                beta  =  -a11*a21-a21*a22 -a22**2-c2/2
                old_s2_approx = [un, vn, alpha*pn + beta * pnm1] # here the value of p is the pressure approxmation
            #=====================
            # third order approx
            #=====================
            elif guess =="third":
                c2 = a22 + a21
                alpha = a11**2*a21 + a11*a21*a22 + 2*a11*a21 + a21*a22**2 + 2* a21*a22 + a22**3 + 2*a22**2 + 11*c2/6
                beta = -2*a11**2*a21 -2*a11*a21*a22 - 3* a11*a21 -2*a21*a22**2 -3*a21*a22 -2*a22**3 -3*a22**2 - 7*c2/6
                gamma = a11**2*a21 + a11*a21*a22 +a11*a21 + a21*a22**2 + a21*a22 + a22**3 + a22**2 + c2/3
                old_s2_approx = [un, vn, alpha * pn + beta * pnm1+ gamma * pnm2]  # here the value of p is the pressure approxmation
            #=====================
            residuals_s2_approx = [f1_s2_approx, f2_s2_approx]
            guesses_s2_approx = [un, vn]
            Tol_s2 = Tol

            sol2, iterations, error, info_2_approx = f.Newton_solver(guesses_s2_approx, old_s2_approx, residuals_s2_approx,
                                                      [f.periodic_u, f.periodic_v, f.periodic_scalar],
                                                       Jacobian=Jacobian_s2_approx,
                                                      Tol=Tol_s2, verbose=False)

            u2, v2 = sol2
            p2 = old_s2_approx[-1]
            info_stage_2 = info_2_approx

        f.periodic_u(u2)
        f.periodic_v(v2)
        f.periodic_scalar(p2)

        rhs_u2 = f.urhs(u2, v2)  # - f.Gpx(p1)
        rhs_v2 = f.vrhs(u2, v2)  # - f.Gpy(p1)

        print('         non-linear iterations at stage 2: ', iterations)
        print('         residual at stage 2: ', error)
        print('         divergence at stage 2: ', np.linalg.norm(f.div(u2, v2).ravel()))


        ######################################
        ######################################
        # stage 3:
        # ---------
        if d3 == 1:

            # unsteady residual stage3
            # -------------------------
            f1_s3 = lambda uold, vold, pold: lambda u, v, p: dt * (
                    (u - uold) / dt - a31 * rhs_u1 - a32 * rhs_u2 - a33 * (f.urhs(u, v) - f.Gpx(p)))
            f2_s3 = lambda uold, vold, pold: lambda u, v, p: dt * (
                    (v - vold) / dt - a31 * rhs_v1 - a32 * rhs_v2 - a33 * (f.vrhs(u, v) - f.Gpy(p)))
            f3_s3 = lambda uold, vold, pold: lambda u, v, p: dt * (
                    -a33 * (-f.div(f.Gpx(p), f.Gpy(p)) + f.div(f.urhs(u, v), f.vrhs(u, v))) \
                    # - a21 * f.div(rhs_u1, rhs_v1) - f.div(uold, vold) / dt)
                    - a31 * f.div(rhs_u1, rhs_v1) - a32 * f.div(rhs_u2, rhs_v2) + (f.div(u, v) - f.div(uold, vold)) / dt)

            old_s3 = [un, vn, pn]
            residuals_s3 = [f1_s3, f2_s3, f3_s3]
            guesses_s3 = [un, vn, pn]
            Tol_s3 = Tol

            sol3, iterations, error, info_3 = f.Newton_solver(guesses_s3, old_s3, residuals_s3,
                                                              [f.periodic_u, f.periodic_v, f.periodic_scalar],
                                                              Jacobian=Jacobian_s3,
                                                              Tol=Tol_s3, verbose=False)
            u3, v3, p3 = sol3
            info_stage_3 = info_3
        elif d3 == 0:
            # unsteady residual stage2
            # -------------------------
            f1_s3_approx = lambda uold, vold, pold: lambda u, v: dt * (
                    (u - uold) / dt - a31 * rhs_u1 - a32 * rhs_u2 - a33 * f.urhs(u, v) + f.Gpx(pold))
            f2_s3_approx = lambda uold, vold, pold: lambda u, v: dt * (
                    (v - vold) / dt - a31 * rhs_v1 - a32 * rhs_v2 - a33 * f.vrhs(u, v) + f.Gpy(pold))

            # second order approx
            # ====================
            if guess == "second":
                c3 = a31 + a32 + a33
                alpha =  a11*a31+a21*a32+a22*a32+a31*a33+a32*a33+a33**2+ 3*c3/2
                beta = -(a11*a31+a21*a32+a22*a32+a31*a33+a32*a33+a33**2+   c3/2)
                old_s3_approx = [un, vn, alpha * pn + beta * pnm1]  # here the value of p is the pressure approxmation
            #======================
            # third order approx
            #======================
            elif guess =="third":
                c3 = a31 + a32 + a33
                alpha = a11**2 * a31 + a11*a21*a32 + a11*a31*a33 + 2*a11*a31 + a21*a22*a32 + a21*a32*a33 + 2*a21*a32 + a22**2*a32 + a22*a32*a33+ 2* a22*a32 + a31*a33**2 + 2*a31*a33 +a32*a33**2+ 2*a32*a33 +a33**3+2*a33**2 + 11*c3/6
                beta = -2*a11**2*a31 - 2*a11*a21*a32 - 2* a11*a31*a33 - 3*a11*a31 - 2*a21*a22*a32 -2*a21*a32*a33 -3*a21*a32 -2*a22**2*a32 -2*a22*a32*a33 -3*a22*a32 - 2*a31*a33**2 - 3*a31*a33 -2*a32*a33**2 - 3*a32*a33 -2*a33**3 - 3*a33**2 -7*c3/6
                gamma = a11**2*a31 + a11*a21*a32 + a11*a31*a33 + a11*a31 + a21*a22*a32 +a21*a32*a33 + a21*a32 + a22**2*a32 +a22*a32*a33+ a22*a32 + a31*a33**2 + a31*a33 + a32*a33**2+ a32*a33 + a33**3 + a33**2 +c3/3
                old_s3_approx = [un, vn, alpha * pn + beta * pnm1 + gamma * pnm2]  # here the value of p is the pressure approxmation
            #======================
            residuals_s3_approx = [f1_s3_approx, f2_s3_approx]
            guesses_s3_approx = [un, vn]
            Tol_s3 = Tol

            sol3, iterations, error, info_3_approx = f.Newton_solver(guesses_s3_approx, old_s3_approx,
                                                                     residuals_s3_approx,
                                                                     [f.periodic_u, f.periodic_v, f.periodic_scalar],
                                                                     Jacobian=Jacobian_s3_approx,
                                                                     Tol=Tol_s3, verbose=False)

            u3, v3 = sol3
            p3 = old_s3_approx[-1]
            info_stage_3 = info_3_approx

        f.periodic_u(u3)
        f.periodic_v(v3)
        f.periodic_scalar(p3)

        info_resid[count] = {'stage_1': info_stage_1, 'stage_2': info_stage_2, 'stage_3':info_stage_3}

        print('         non-linear iterations at stage 3: ', iterations)
        print('         residual at stage 3: ', error)
        print('         divergence at stage 3: ', np.linalg.norm(f.div(u2, v2).ravel()))


        ######################################
        ######################################



        # time n+1
        #----------
        uhnp1 = un + b1 * dt * f.urhs(u1,v1) + b2 * dt * f.urhs(u2, v2) + b3 * dt * f.urhs(u3, v3)
        vhnp1 = vn + b1 * dt * f.vrhs(u1,v1) + b2 * dt * f.vrhs(u2, v2) + b3 * dt * f.vrhs(u3, v3)

        unp1, vnp1, press, iter = f.ImQ(uhnp1, vhnp1, Coef, pn)

        f.periodic_u(unp1)
        f.periodic_v(vnp1)
        f.periodic_scalar(press)

        time_end = time.time()

        if post_projection:
            # post processing projection
            uhnp1_star = un + dt * (f.urhs(unp1, vnp1))
            vhnp1_star = vn + dt * (f.vrhs(unp1, vnp1))

            _, _, post_press, _ = f.ImQ(uhnp1_star, vhnp1_star, Coef, pn)

        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        print('        Mass residual:',residual)

        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        print('len Usol:',len(usol))
        print('Courant Number=',1.42*dt/dx)
        # iterations.append(iter)
        # # u and v num cell centered
        ucc = 0.5*(un[1:-1,2:] + un[1:-1,1:-1])
        vcc = 0.5*(vn[2:,1:-1] + vn[1:-1,1:-1])
        #
        uexc = uexact(a,b,xu,yu,t)
        vexc = vexact(a,b,xv,yv,t)
        # u and v exact cell centered
        uexc_cc = 0.5*(uexc[:,:-1] + uexc[:,1:])
        vexc_cc = 0.5*(vexc[:-1,:] + vexc[1:,:])
        t += dt

        # compute of kinetic energy
        ken_new = np.sum(ucc.ravel()**2 +vcc.ravel()**2)/2
        ken_exact = np.sum(uexc_cc.ravel()**2 +vexc_cc.ravel()**2)/2
        print('time = ',t)
        print('ken_new = ',ken_new)
        print('target_ken=', target_ke)
        print('ken_exc = ',ken_exact)
        print('(ken_new - ken_old)/ken_old = ',(ken_new - ken_old)/ken_old)
        if (((ken_new - ken_old)/ken_old) > 0 and count>1) or np.isnan(ken_new):
            is_stable = False
            print('is_stable = ',is_stable)
            if stability_counter >3:
                print('not stable !!!!!!!!')
                break
            else:
                stability_counter += 1
        else:
            is_stable = True
            print('is_stable = ', is_stable)
            if ken_new < target_ke and count > 30:
                break
        ken_old = ken_new.copy()
        print('is_stable = ', is_stable)
        # CFL = 0.5
        # Reh = 1.42 * dx / μ
        # im = plt.imshow((psol[-1][1:-1, 1:] - psol[-1][1:-1, :-1]) / dx, origin="bottom")
        # plt.title("Reh={:0.4f} time={:0.4f}s".format(Reh, t))
        # plt.tight_layout()
        # # plt.savefig(
        # #     'analytical_J_Implicit_NSE/DIRK2_pr_theta_0.5/convergence/Pyamg/CFL-{}/gpx/timestep-{:0>2}.png'.format(
        # #         CFL, count), dpi=400)
        # # plt.close()
        # plt.show()
        #
        # im = plt.imshow(usol[-1][1:-1, 1:], origin="bottom")
        # plt.title("Reh={:0.4f} time={:0.4f}s".format(Reh, t))
        # plt.tight_layout()
        # plt.savefig(
        #     'analytical_J_Implicit_NSE/DIRK2_pr_theta_0.5/convergence/Pyamg/CFL-{}/unp1/timestep-{:0>2}.png'.format(
        #         CFL, count), dpi=400)
        # plt.close()

        count+=1
    diff = np.linalg.norm(uexact(a,b,xu,yu,t).ravel()-unp1[1:-1,1:] .ravel(),np.inf)
    if return_stability:
        return is_stable
    else:
        return diff, [total_iteration], is_stable, unp1[1:-1, 1:].ravel(), info_resid


# from singleton_classes import ProbDescription
# import matplotlib.pyplot as plt
# #
# dt_lam = lambda CFL, dx,Uinlet: CFL*dx/Uinlet
# Uinlet = 1.42
# ν = 0.1
#
# probDescription = ProbDescription(N=[32,32],L=[1,1],μ =ν,dt = 0.01)
# dx,dy = probDescription.dx, probDescription.dy
# # dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# dt = dt_lam(0.5,dx,Uinlet)
# probDescription.set_dt(dt)
# _,_,_,_,resid_info = DIRK2_taylor_vortex(steps=20, return_stability=False,name='pr',guess='DIRK2', project=[1,1],alpha=0.99,theta=0.5)
#
# # resid_filename ='analytical_J_Implicit_NSE/DIRK2_pr_theta_0.5/convergence/Pyamg/CFL-0.5/residuals_per_timestep.json'
# # with open(resid_filename,"w") as file:
# #     json.dump(resid_info,file,indent=4)