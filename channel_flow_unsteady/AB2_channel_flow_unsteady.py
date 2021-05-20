import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc


def AB2_channel_flow_unsteady(steps=3, return_stability=False, name='heun',alpha=0.9,theta=None,post_projection=False):
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
    np.random.seed(123)
    mag = 0
    u0 = np.random.rand(ny + 2, nx + 2) * mag - 0.5 * mag  # include ghost cells
    # u0 = np.zeros([ny +2, nx+2])# include ghost cells
    # same thing for the y-velocity component
    v0 = np.random.rand(ny + 2, nx + 2) * mag - 0.5 * mag  # include ghost cells
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
    v_bc_left_wall = lambda t: lambda yv: np.sin(at(t))

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

    u0_free, v0_free, _, _ = f.ImQ_bcs(u0, v0, Coef, np.zeros([nx + 2, ny + 2]), p_bcs)

    f.top_wall(u0_free, v0_free, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0_free, v0_free, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0_free, v0_free, u_bc_right_wall(u0_free[1:-1, -1]), v_bc_right_wall(v0_free[1:, -2]))
    f.left_wall(u0_free, v0_free, u_bc_left_wall(t), v_bc_left_wall(t))

    # initialize the pressure
    p0 = np.zeros([nx + 2, ny + 2]);  # include ghost cells

    # declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1 = np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol = []
    # usol.append(u0)
    usol.append(u0_free)
    # usol.append(u_FE)
    #
    vsol = []
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
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 = np.zeros_like(u)
        # pnm2 = np.zeros_like(u)
        if count <= 1: # Starting the Adam-Bashforth with RK2 for the first two steps.
            # rk coefficients
            RK2 = sc.RK2(name, theta)
            a21 = RK2.a21
            b1 = RK2.b1
            b2 = RK2.b2

            ## stage 1
            print('    Stage 1:')
            print('    --------')
            time_start = time.clock()
            u1 = u.copy()
            v1 = v.copy()

            # Au1

            # apply boundary conditions before the computation of the rhs
            f.top_wall(u1, v1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(u1, v1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(u1, v1, u_bc_right_wall(u1[1:-1, -1]),
                         v_bc_right_wall(v1[1:, -2]))  # this won't change anything for u2
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
            f.right_wall(uh2, vh2, u_bc_right_wall(uh2[1:-1, -2]),
                         v_bc_right_wall(vh2[1:, -1]))  # this won't change anything for u2
            f.left_wall(uh2, vh2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

            press_stage_2 = np.zeros([nx + 2, ny + 2])
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, press_stage_2, iter1 = f.ImQ_bcs(uh2, vh2, Coef, pn, p_bcs)
            iteration_i_2 += iter1
            print('        iterations stage 2 = ', iter1)
            # apply bcs
            f.top_wall(u2, v2, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(u2, v2, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(u2, v2, u_bc_right_wall(u2[1:-1, -1]),
                         v_bc_right_wall(v2[1:, -2]))  # this won't change anything for u2
            f.left_wall(u2, v2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

            div2 = np.linalg.norm(f.div(u2, v2).ravel())
            print('        divergence of u2 = ', div2)
            urhs2 = f.urhs_bcs(u2, v2)
            vrhs2 = f.vrhs_bcs(u2, v2)

            uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2)
            vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2)

            f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]),
                         v_bc_right_wall(vhnp1[1:, -1]))  # this won't change anything for u2
            f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

            # unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn,p_bcs)
            unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, press_stage_2, p_bcs)

            iteration_np1 += iter2
            print('iter_np1=', iter2)

            # apply bcs
            f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]),
                         v_bc_right_wall(vnp1[1:, -2]))  # this won't change anything for unp1
            f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        elif count > 1:
            # switch to two step Adam Bashforth
            print("Adam-Bashforth 2 Steps")
            print("----------------------")
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()

            un = u.copy()
            vn = v.copy()

            unm1 = usol[-2].copy()
            vnm1 = vsol[-2].copy()

            urhsn = f.urhs(un, vn)
            vrhsn = f.vrhs(un, vn)

            urhsnm1 = f.urhs(unm1, vnm1)
            vrhsnm1 = f.vrhs(unm1, vnm1)

            b1 = 3.0/2
            b2 = -1.0/2
            # Adam Bashforth two step method
            uhnp1 = un + dt * ( b1 * (urhsn) + b2 * (urhsnm1))
            vhnp1 = vn + dt * ( b1 * (vrhsn) + b2 * (vrhsnm1))

            f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]),
                         v_bc_right_wall(vhnp1[1:, -1]))  # this won't change anything for u2
            f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

            unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn, p_bcs)

            f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]),
                         v_bc_right_wall(vnp1[1:, -2]))  # this won't change anything for unp1
            f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        if post_projection:
            # post processing projection
            uhnp1_star = u + dt * (f.urhs(unp1, vnp1))
            vhnp1_star = v + dt * (f.vrhs(unp1, vnp1))

            f.top_wall(uhnp1_star, vhnp1_star, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(uhnp1_star, vhnp1_star, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(uhnp1_star, vhnp1_star, u_bc_right_wall(uhnp1_star[1:-1, -2]),
                         v_bc_right_wall(vhnp1_star[1:, -1]))
            f.left_wall(uhnp1_star, vhnp1_star, u_bc_left_wall(t+2*dt), v_bc_left_wall(t+2*dt))

            _, _, post_press, _ = f.ImQ_bcs(uhnp1_star, vhnp1_star, Coef, pn, p_bcs)

        new_press = (3 * press - pn) / 2  # second order

        iteration_np1+=iter2

        time_end = time.clock()
        psol.append(press)
        cpu_time = time_end - time_start
        print('cpu_time=', cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('Mass residual:', residual)
        print('iterations last stage:', iter2)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)


        # plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        gradpx = (psol[-1][1:-1, 1:] - psol[-1][1:-1, :-1]) / dx

        maxbound = max(gradpx[1:-1, 1:].ravel())
        minbound = min(gradpx[1:-1, 1:].ravel())
        # plot of the pressure gradient in order to make sure the solution is correct
        # if count%1 == 0:
        #     # plt.imshow(gradpx[1:-1,1:],origin='bottom',cmap='jet',vmax=maxbound, vmin=minbound)
        #     plt.imshow(unp1[1:-1,1:],origin='bottom',cmap='jet',vmax=maxbound, vmin=minbound)
        #     # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        #     v = np.linspace(minbound, maxbound, 4, endpoint=True)
        #     plt.colorbar(ticks=v)
        #     plt.title('time: {}'.format(t))
        #     plt.show()
        count += 1
    if return_stability:
        return True
    else:
        return True, [iteration_i_2, iteration_np1], True, new_press[1:-1, 1:-1].ravel()

# import matplotlib.pyplot as plt
# from core.singleton_classes import ProbDescription
#
# probDescription = ProbDescription(N=[128,16],L=[4,1],μ =0.01,dt = 0.001)
# AB2_channel_flow_unsteady(steps=100, return_stability=False, name='heun',alpha=0.9,theta=None,post_projection=False)