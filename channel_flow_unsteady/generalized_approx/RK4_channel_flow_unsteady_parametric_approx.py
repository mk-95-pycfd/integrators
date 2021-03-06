import numpy as np
from core.functions import func
import time
import core.singleton_classes as sc
import statistics
import pyamg
import matplotlib.pyplot as plt

def RK4_channel_flow_unsteady_parametric_approx (steps = 3,return_stability=False, name='heun', guess=None, project=[],gamma={
    "22":0,"23":0,"33":0},alpha=0.99,post_projection=False,user_bcs_time_func=[],eliminate_extra_terms_in_phi2=False):
    probDescription = sc.ProbDescription()
    f = func(probDescription)
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()

    t = 0.0
    tend = steps
    count = 0
    print('dt=', dt)
    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    # np.random.seed(123)
    # u0 = np.random.rand(ny + 2, nx + 2)/1e7  # include ghost cells
    u0 = np.zeros([ny +2, nx+2])# include ghost cells
    # same thing for the y-velocity component
    # v0 = np.random.rand(ny + 2, nx + 2)/1e7  # include ghost cells
    v0 = np.zeros([ny +2, nx+2])  # include ghost cells

    # ---------------------------------------------
    # Definition of the time dependent ubc
    # ---------------------------------------------
    field = np.zeros_like(u0)
    xu_, yu_ = probDescription.get_XVol()
    field[1:-1, 1] = yu_[:, 0] * (1.0 - yu_[:, 0])
    if user_bcs_time_func:
        print("using user defined time function")
        time_func, time_func_prime = user_bcs_time_func
    else:
        time_func = lambda t: 1
        time_func_prime = lambda t: 0
    ub = lambda t: field * time_func(t)
    ub_prime = lambda t: field * time_func_prime(t)

    def extra_phi2(t, c2):
        print('************* solving for the extra terms in phi2 ***************')
        rhs = (ub_prime(t) - (ub(t + c2 * dt) - ub(t)) / c2 / dt) / dx
        ml = pyamg.ruge_stuben_solver(Coef)
        tmpsol = ml.solve(np.ravel(rhs[1:-1, 1:-1]), tol=1e-16)
        sol = np.zeros([ny + 2, nx + 2])
        sol[1:-1, 1:-1] = tmpsol.reshape([ny, nx])
        return sol

    # ==============================================

    u_bc_top_wall = lambda xu: 0
    u_bc_bottom_wall = lambda xu: 0
    u_bc_right_wall = lambda u: lambda yu: u
    u_bc_left_wall = lambda t: lambda yu: yu_[:, 0] * (1.0 - yu_[:, 0]) * time_func(t) # time dependent parabolic inlet

    v_bc_top_wall = lambda xv: 0
    v_bc_bottom_wall = lambda xv: 0
    v_bc_right_wall = lambda v: lambda yv: v
    v_bc_left_wall = lambda t: lambda yv: 0

    # pressure
    def pressure_right_wall(p):
        # pressure on the right wall
        p[1:-1, -1] = -p[1:-1, -2]

    p_bcs = lambda p: pressure_right_wall(p)
    # apply bcs
    f.top_wall(u0, v0, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0, v0, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0, v0, u_bc_right_wall(u0[1:-1, -2]), v_bc_right_wall(v0[1:, -1]))
    f.left_wall(u0, v0, u_bc_left_wall(t), v_bc_left_wall(t))

    Coef = f.A_channel_flow()

    # u0_free, v0_free, press0_free, _ = f.ImQ_bcs(u0, v0, Coef, 0, p_bcs)
    u0_free = np.zeros_like(u0)
    v0_free = np.zeros_like(v0)

    divuhat = f.div(u0, v0)
    prhs = divuhat[1:-1, 1:-1]
    rhs = prhs.ravel()
    ml = pyamg.ruge_stuben_solver(Coef)
    ptmp = ml.solve(rhs, tol=1e-16)
    press0_free = np.zeros([ny + 2, nx + 2])
    press0_free[1:-1, 1:-1] = ptmp.reshape([ny, nx])

    # set bcs on the pressure
    p_bcs(press0_free)

    u0_free[1:-1, 1:] = u0[1:-1, 1:] -  (press0_free[1:-1, 1:] - press0_free[1:-1, :-1]) / dx
    v0_free[1:, 1:-1] = v0[1:, 1:-1] -  (press0_free[1:, 1:-1] - press0_free[:-1, 1:-1]) / dy

    f.top_wall(u0_free, v0_free, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0_free, v0_free, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0_free, v0_free, u_bc_right_wall(u0_free[1:-1, -1]), v_bc_right_wall(v0_free[1:, -2]))
    f.left_wall(u0_free, v0_free, u_bc_left_wall(t), v_bc_left_wall(t))

    print('div_u0=', np.linalg.norm(f.div(u0_free, v0_free).ravel()))

    # initialize the pressure
    p0 = np.zeros([ny + 2, nx + 2]);  # include ghost cells

    # declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1 = np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol = []
    # usol.append(u0)
    usol.append(u0_free)

    vsol = []
    # vsol.append(v0)
    vsol.append(v0_free)

    psol = []
    psol.append(press0_free)
    iterations = [0]

    while count < tend:
        # plt.contourf(xcc, ycc, psol[-1][1:-1, 1:-1])
        # plt.colorbar()
        # plt.title('pn time:{}'.format(t))
        # plt.show()
        RK4 = sc.RK4(name)
        a21 = RK4.a21
        a31 = RK4.a31
        a32 = RK4.a32
        a41 = RK4.a41
        a42 = RK4.a42
        a43 = RK4.a43
        b1 = RK4.b1
        b2 = RK4.b2
        b3 = RK4.b3
        b4 = RK4.b4
        print('timestep:{}'.format(count + 1))
        print('-----------')
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 = np.zeros_like(u)
        pnm2 = np.zeros_like(u)
        pnm3 = np.zeros_like(u)
        if count > 4:
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            pnm2 = psol[-3].copy()
            pnm3 = psol[-4].copy()

            d2, d3, d4 = project
            # define the pressure derivatives
            Pn = 11 / 6 * pn - 7 / 6 * pnm1 + pnm2 / 3
            Pn_p = (2 * pn - 3 * pnm1 + pnm2) / dt
            Pn_pp = (pn - 2 * pnm1 + pnm2) / dt / dt

            c3 = a31 + a32
            c4 = a41 + a42 + a43
            # define parameters

            # dependent on the parameters
            gamma_22 = gamma["22"] #a21/2
            gamma_23 = gamma["23"] #a21**2/6
            gamma_33 = gamma["33"] #c3**2/6
            #---------------------------------

            gamma_32 = (1/24-gamma_22*(a21*a32*b3+a21*a42*b4))/(a43 * b4 * c3)
            gamma_42 = -(a21*b2*gamma_22)/(b4*c4)-(b3 * gamma_32*c3)/(b4*c4)+1/(6*b4*c4)
            gamma_43 = -(gamma_23*(a21*b2))/(b4*c4)-(gamma_33*(b3*c3))/(b4*c4)+1/(24*b4*c4)

            print("general approx")
            print("gamma_22=", gamma_22)
            print("gamma_23=", gamma_23)
            print("gamma_32=", gamma_32)
            print("gamma_33=", gamma_33)
            print("gamma_42=", gamma_42)
            print("gamma_43=", gamma_43)
            print("==================")
            # comment on capuano paper
            # gamma_22 = a21/2
            # gamma_23 = a21**2/6
            # gamma_32 = c3/2
            # gamma_33 = c3**2/6
            # gamma_42 = c4/2
            # gamma_43 = c4**2/6

            # print("capuano's approx")
            # print("gamma_22=", a21/2)
            # print("gamma_23=", a21**2/6)
            # print("gamma_32=", c3/2)
            # print("gamma_33=", c3**2/6)
            # print("gamma_42=", c4/2)
            # print("gamma_43=", c4**2/6)

        elif count <= 4:  # compute pressures for 4 time steps
            d2 = 1
            d3 = 1
            d4 = 1
        print('    Stage 1:')
        print('    --------')
        time_start = time.time()
        u1 = u.copy()
        v1 = v.copy()

        f.top_wall(u1, v1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u1, v1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u1, v1, u_bc_right_wall(u1[1:-1, -1]),
                     v_bc_right_wall(v1[1:, -2]))  # this won't change anything for u2
        f.left_wall(u1, v1, u_bc_left_wall(t), v_bc_left_wall(t))

        # Au1
        urhs1 = f.urhs_bcs(u1, v1)
        vrhs1 = f.vrhs_bcs(u1, v1)

        # divergence of u1
        div_n = np.linalg.norm(f.div(u1, v1).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh2 = u + a21 * dt * urhs1
        vh2 = v + a21 * dt * vrhs1

        f.top_wall(uh2, vh2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uh2, vh2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uh2, vh2, u_bc_right_wall(uh2[1:-1, -2]),v_bc_right_wall(vh2[1:, -1]))
        f.left_wall(uh2, vh2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

        if d2 == 1:
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, phi2, iter1 = f.ImQ_bcs(uh2, vh2, Coef, pn, p_bcs)
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            if guess == "third":
                p_approx = Pn + gamma_22 * dt * Pn_p + gamma_23 * dt * dt * Pn_pp
            else:
                p_approx = np.zeros_like(pn)

            u2 = uh2 - a21 * dt * f.Gpx(p_approx)
            v2 = vh2 - a21 * dt * f.Gpy(p_approx)

        # apply bcs
        f.top_wall(u2, v2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u2, v2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u2, v2, u_bc_right_wall(u2[1:-1, -1]), v_bc_right_wall(v2[1:,-2]))  # this won't change anything for u2
        f.left_wall(u2, v2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)
        urhs2 = f.urhs_bcs(u2, v2)
        vrhs2 = f.vrhs_bcs(u2, v2)

        ## stage 3
        print('    Stage 3:')
        print('    --------')
        uh3 = u + a31 * dt * urhs1 + a32 * dt * urhs2
        vh3 = v + a31 * dt * vrhs1 + a32 * dt * vrhs2

        f.top_wall(uh3, vh3, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uh3, vh3, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uh3, vh3, u_bc_right_wall(uh3[1:-1, -2]), v_bc_right_wall(vh3[1:,-1]))
        f.left_wall(uh3, vh3, u_bc_left_wall(t+(a31+a32)*dt), v_bc_left_wall(t+(a31+a32)*dt))

        if d3 == 1:
            print('        pressure projection stage{} = True'.format(3))
            u3, v3, _, iter1 = f.ImQ_bcs(uh3, vh3, Coef, pn, p_bcs)
            print('        iterations stage 3 = ', iter1)
        elif d3 == 0:
            if guess == "third":
                p_approx = Pn + gamma_32 * dt * Pn_p + gamma_33 * dt * dt * Pn_pp
            else:
                p_approx = np.zeros_like(pn)

            u3 = uh3 - (a31 + a32) * dt * f.Gpx(p_approx)
            v3 = vh3 - (a31 + a32) * dt * f.Gpy(p_approx)

        # apply bcs
        f.top_wall(u3, v3, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u3, v3, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u3, v3, u_bc_right_wall(u3[1:-1, -1]), v_bc_right_wall(v3[1:,-2]))  # this won't change anything for u2
        f.left_wall(u3, v3, u_bc_left_wall(t+(a31+a32)*dt), v_bc_left_wall(t+(a31+a32)*dt))

        div3 = np.linalg.norm(f.div(u3, v3).ravel())
        print('        divergence of u3 = ', div3)
        urhs3 = f.urhs_bcs(u3, v3)
        vrhs3 = f.vrhs_bcs(u3, v3)

        ## stage 4
        print('    Stage 4:')
        print('    --------')
        uh4 = u + a41 * dt * urhs1 + a42 * dt * urhs2 + a43 * dt * urhs3
        vh4 = v + a41 * dt * vrhs1 + a42 * dt * vrhs2 + a43 * dt * vrhs3

        f.top_wall(uh4, vh4, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uh4, vh4, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uh4, vh4, u_bc_right_wall(uh4[1:-1, -2]), v_bc_right_wall(vh4[1:, -1]))
        f.left_wall(uh4, vh4, u_bc_left_wall(t + (a41 + a42 + a43) * dt), v_bc_left_wall(t + (a41 + a42 + a43) * dt))

        if d4 == 1:
            print('        pressure projection stage{} = True'.format(3))
            u4, v4, _, iter1 = f.ImQ_bcs(uh4, vh4, Coef, pn, p_bcs)
            print('        iterations stage 3 = ', iter1)
        elif d4 == 0:
            if guess == "third":
                p_approx = Pn + gamma_42 * dt * Pn_p + gamma_43 * dt * dt * Pn_pp
            else:
                p_approx = np.zeros_like(pn)

            u4 = uh4 - (a41 + a42 + a43) * dt * f.Gpx(p_approx)
            v4 = vh4 - (a41 + a42 + a43) * dt * f.Gpy(p_approx)

        # apply bcs
        f.top_wall(u4, v4, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u4, v4, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u4, v4, u_bc_right_wall(u4[1:-1, -1]),
                     v_bc_right_wall(v4[1:, -2]))  # this won't change anything for u2
        f.left_wall(u4, v4, u_bc_left_wall(t + (a41 + a42 + a43) * dt), v_bc_left_wall(t + (a41 + a42 + a43) * dt))

        div4 = np.linalg.norm(f.div(u4, v4).ravel())
        print('        divergence of u4 = ', div4)
        urhs4 = f.urhs_bcs(u4, v4)
        vrhs4 = f.vrhs_bcs(u4, v4)

        uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2) + dt * b3 * (urhs3) + dt * b4 * (urhs4)
        vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2) + dt * b3 * (vrhs3) + dt * b4 * (vrhs4)

        f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]),v_bc_right_wall(vhnp1[1:,-1]))  # this won't change anything for unp1
        f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn, p_bcs)

        # apply bcs
        f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]),v_bc_right_wall(vnp1[1:,-2]))  # this won't change anything for unp1
        f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        time_end = time.time()

        # if post_projection:
        #     # post processing projection
        #     mp_field = np.zeros_like(unp1)
        #     mp_field[1:-1, 1] = m_p(t + dt, yu[:, 0])
        #
        #     unp1_rhs = f.urhs_bcs(unp1, vnp1)
        #     vnp1_rhs = f.vrhs_bcs(unp1, vnp1)
        #     rhs_pp = np.zeros_like(pn)
        #     rhs_pp = f.div(unp1_rhs, vnp1_rhs) #- mp_field
        #
        #     ml = pyamg.ruge_stuben_solver(Coef)
        #     ptmp = ml.solve(np.ravel(rhs_pp[1:-1, 1:-1]), tol=1e-12)
        #
        #     nx = probDescription.nx
        #     ny = probDescription.ny
        #     post_press = np.zeros([ny + 2, nx + 2])
        #     post_press[1:-1, 1:-1] = ptmp.reshape([ny, nx])
        #     p_bcs(post_press)

        # # --------------------------------------
        # # construction of instantaneous pressure
        # # --------------------------------------
        # alpha_num = 1 / 2
        # beta = -2
        # gamma = 5 / 2
        # new_press = alpha_num * pn + beta * phi2 + gamma * press
        # if user_bcs_time_func and eliminate_extra_terms_in_phi2:
        #     q2 = extra_phi2(t, a21)
        #     new_press -= beta*q2
        # # ======================================

        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=', cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:', residual)
        print('iterations:', iter)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        iterations.append(iter)

        t += dt

        # plot of the pressure gradient in order to make sure the solution is correct
        # plt.contourf(usol[-1][1:-1,1:])
        # if count % 1 ==0:
        #     # divu = f.div(unp1,vnp1)
        #     # plt.imshow(divu[1:-1,1:-1], origin='bottom')
        #     # plt.colorbar()
        #     ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        #     vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])
        #     speed = np.sqrt(ucc * ucc + vcc * vcc)
        #     # uexact = 4 * 1.5 * ycc * (1 - ycc)
        #     # plt.plot(uexact, ycc, '-k', label='exact')
        #     # plt.plot(ucc[:, int(8 / dx)], ycc, '--', label='x = {}'.format(8))
        #     plt.contourf(xcc, ycc, new_press[1:-1,1:-1])
        #     # plt.contourf(xcc, ycc, speed)
        #     plt.colorbar()
        #     plt.title('pinst time:{}'.format(t))
        #     # plt.streamplot(xcc, ycc, ucc, vcc, color='black', density=0.75, linewidth=1.5)
        #     # plt.contourf(xcc, ycc, psol[-1][1:-1, 1:-1])
        #     # plt.colorbar()
        #     plt.show()
        count += 1

    if return_stability:
        return True
    else:
        return True, [div_np1], True, unp1[1:-1, 1:-1].ravel()


# from core.singleton_classes import ProbDescription
# #
# Uinlet = 1
# ν = 0.01
# probDescription = ProbDescription(N=[4*32,32],L=[4,1],μ =ν,dt = 0.0025)
# dx,dy = probDescription.dx, probDescription.dy
# # dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# # probDescription.set_dt(dt)
# RK3_channel_flow_unsteady (steps = 2000,return_stability=False, name='regular', guess=None, project=[1,1],alpha=0.99)