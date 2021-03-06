import numpy as np
from core.functions import func
import time
import core.singleton_classes as sc
import pyamg

def RK2_channel_flow_unsteady (steps = 3,return_stability=False, name='heun', guess=None, project=[],theta=None,post_projection=False):
    probDescription = sc.ProbDescription()
    f = func(probDescription)
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()

    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)
    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    np.random.seed(123)
    mag = 0
    u0 = np.random.rand(ny + 2, nx + 2)*mag - 0.5*mag # include ghost cells
    # u0 = np.zeros([ny +2, nx+2])# include ghost cells
    # same thing for the y-velocity component
    v0 = np.random.rand(ny + 2, nx + 2)*mag -0.5*mag # include ghost cells
    # v0 = np.zeros([ny +2, nx+2])  # include ghost cells

    Min = np.sum(np.ones_like(u0[1:ny + 1, 1]))

    at = lambda t: (np.pi / 6) * np.sin(t / 2)

    u_bc_top_wall = lambda xu: 0
    u_bc_bottom_wall = lambda xu: 0
    u_bc_right_wall = lambda u: lambda yu: u
    u_bc_left_wall = lambda t: lambda yu: 4.0 * yu[:, 0] * (1.0 - yu[:, 0]) * np.cos(at(t))  # parabolic inlet
    # u_bc_left_wall = lambda yu: 1.0 # uniform inlet

    v_bc_top_wall = lambda xv: 0
    v_bc_bottom_wall = lambda xv: 0
    v_bc_right_wall = lambda v: lambda yv: v
    v_bc_left_wall = lambda t: lambda yv: 0

    # post processing m'
    # we only consider the u component of the velocity (check sanderse 2012)
    m_p = lambda t, y: -np.pi * y * (1 - y) * np.sin(at(t)) * np.cos(t / 2.0)

    # pressure
    def pressure_right_wall(p):
        # pressure on the right wall
        p[1:-1, -1] = -p[1:-1, -2]

    p_bcs = lambda p: pressure_right_wall(p)
    # apply bcs
    f.top_wall(u0, v0, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0, v0, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0, v0, u_bc_right_wall(u0[1:-1, -2]), v_bc_right_wall(v0[1:,-1]))
    f.left_wall(u0, v0, u_bc_left_wall(t), v_bc_left_wall(t))

    Coef = f.A_channel_flow()

    u0_free, v0_free, _, _ = f.ImQ_bcs(u0, v0, Coef, np.zeros([nx+2,ny+2]), p_bcs)

    f.top_wall(u0_free, v0_free, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0_free, v0_free, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0_free, v0_free, u_bc_right_wall(u0_free[1:-1, -1]), v_bc_right_wall(v0_free[1:,-2]))
    f.left_wall(u0_free, v0_free, u_bc_left_wall(t), v_bc_left_wall(t))

    # initialize the pressure
    p0 = np.zeros([nx+2,ny+2]); # include ghost cells

    #declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1= np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol=[]
    # usol.append(u0)
    usol.append(u0_free)
    # usol.append(u_FE)
    #
    vsol=[]
    # vsol.append(v0)
    vsol.append(v0_free)
    # vsol.append(v_FE)

    psol = []
    psol.append(p0)
    iteration_i_2 = 0
    iteration_np1 = 0

    while count < tend:
        print('timestep:{}'.format(count + 1))
        print('-----------')
        # rk coefficients
        RK2 = sc.RK2(name,theta)
        a21 = RK2.a21
        b1 = RK2.b1
        b2 = RK2.b2
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 = np.zeros_like(u)
        # pnm2 = np.zeros_like(u)# only needed for high accurate pressure
        if count > 1: # change the count for 2 if high accurate pressure at time np1 is needed
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            # pnm2 = psol[-3].copy() # only needed for high accurate pressure
            d2, = project

        elif count <= 1:  # compute pressures for 2 time steps # change the count for 2 if high accurate pressure at time np1 is needed
            d2 = 1
            iteration_i_2 = 0
            iteration_np1 = 0
        ## stage 1

        print('    Stage 1:')
        print('    --------')
        time_start = time.time()
        u1 = u.copy()
        v1 = v.copy()

        # Au1

        # apply boundary conditions before the computation of the rhs
        f.top_wall(u1, v1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u1, v1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u1, v1, u_bc_right_wall(u1[1:-1, -1]), v_bc_right_wall(v1[1:,-2]))  # this won't change anything for u2
        f.left_wall(u1, v1, u_bc_left_wall(t), v_bc_left_wall(t))

        urhs1 = f.urhs_bcs(u1, v1)
        vrhs1 = f.vrhs_bcs(u1, v1)

        # divergence of u1
        div_n = np.linalg.norm(f.div(u1, v1).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh2 = u + a21 * dt * (urhs1)
        vh2 = v + a21 * dt * (vrhs1)

        f.top_wall(uh2, vh2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uh2, vh2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uh2, vh2, u_bc_right_wall(uh2[1:-1, -2]), v_bc_right_wall(vh2[1:,-1]))  # this won't change anything for u2
        f.left_wall(uh2, vh2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

        press_stage_2=np.zeros([nx+2,ny+2])

        if d2 == 1:
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, press_stage_2, iter1 = f.ImQ_bcs(uh2, vh2, Coef, pn,p_bcs)
            iteration_i_2 += iter1
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            if guess == "first":
                p_approx = pn
            else:
                p_approx = 0

            u2 = uh2 - a21 * dt * f.Gpx(p_approx)
            v2 = vh2 - a21 * dt * f.Gpy(p_approx)

        # apply bcs
        f.top_wall(u2, v2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u2, v2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u2, v2, u_bc_right_wall(u2[1:-1, -1]), v_bc_right_wall(v2[1:,-2])) # this won't change anything for u2
        f.left_wall(u2, v2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))


        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)
        urhs2 = f.urhs_bcs(u2, v2)
        vrhs2 = f.vrhs_bcs(u2, v2)

        uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2)
        vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2)

        f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]), v_bc_right_wall(vhnp1[1:,-1]))  # this won't change anything for u2
        f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        # unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn,p_bcs)
        unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, press_stage_2,p_bcs)

        iteration_np1 += iter2
        print('iter_np1=',iter2)

        # apply bcs
        f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]), v_bc_right_wall(vnp1[1:,-2])) # this won't change anything for unp1
        f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        time_end = time.time()

        if post_projection:
            # post processing projection
            mp_field = np.zeros_like(unp1)
            mp_field[1:-1, 1] = m_p(t + dt, yu[:, 0])

            unp1_rhs = f.urhs_bcs(unp1, vnp1)
            vnp1_rhs = f.vrhs_bcs(unp1, vnp1)
            rhs_pp = np.zeros_like(pn)
            rhs_pp = f.div(unp1_rhs, vnp1_rhs) - mp_field

            ml = pyamg.ruge_stuben_solver(Coef)
            ptmp = ml.solve(rhs_pp[1:-1, 1:-1], tol=1e-12)

            nx = probDescription.nx
            ny = probDescription.ny
            post_press = np.zeros([ny + 2, nx + 2])
            post_press[1:-1, 1:-1] = ptmp.reshape([ny, nx])
            p_bcs(post_press)


        new_press =  (3*press -pn) / 2 #second order



        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1,vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:',residual)
        print('iterations:',iter)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)

        Min = np.sum(unp1[1:ny+1,1])
        Mout = np.sum(unp1[1:ny+1,nx+1])

        print("Min=",Min)
        print("Mout=",Mout)


        t += dt

        # plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        # if count % 1 ==0:
        #     # divu = f.div(u0_free,v0_free)
        #     # plt.imshow(divu[1:-1,1:-1], origin='bottom')
        #     # plt.colorbar()
        #     ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        #     vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])
        #     speed = np.sqrt(ucc * ucc + vcc * vcc)
        #     # uexact = 4 * 1.5 * ycc * (1 - ycc)
        #     # plt.plot(uexact, ycc, '-k', label='exact')
        #     # plt.plot(ucc[:, int(8 / dx)], ycc, '--', label='x = {}'.format(8))
        #     plt.contourf(xcc, ycc, speed)
        #     plt.colorbar()
        #     # plt.streamplot(xcc, ycc, ucc, vcc, color='black', density=0.75, linewidth=1.5)
        #     # plt.contourf(xcc, ycc, psol[-1][1:-1, 1:-1])
        #     # plt.colorbar()
        #     plt.show()
        count += 1

    if return_stability:
        return True
    else:
        return True, [iteration_i_2,iteration_np1], True, unp1[1:-1,1:-1].ravel()


# from core.singleton_classes import ProbDescription
# import matplotlib.pyplot as plt
# Uinlet = 1
# ν = 0.01
# probDescription = ProbDescription(N=[4*32,32],L=[10,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt)
# RK2_channel_flow_unsteady (steps = 10,return_stability=False, name='heun', guess=None, project=[1])