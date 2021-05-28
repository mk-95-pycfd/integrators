import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc
import pyamg

def AB3_channel_flow_unsteady(steps=3, return_stability=False, name='heun',alpha=0.9,theta=None,post_projection=False):
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
        if count <= 2: # Starting the Adam-Bashforth with RK2 for the first two steps.
            # rk coefficients
            RK3 = sc.RK3(name)
            a21 = RK3.a21
            a31 = RK3.a31
            a32 = RK3.a32
            b1 = RK3.b1
            b2 = RK3.b2
            b3 = RK3.b3

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

            ## stage 3
            print('    Stage 3:')
            print('    --------')
            uh3 = u + a31 * dt * (urhs1) + a32 * dt * (urhs2)
            vh3 = v + a31 * dt * (vrhs1) + a32 * dt * (vrhs2)

            f.top_wall(uh3, vh3, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(uh3, vh3, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(uh3, vh3, u_bc_right_wall(uh3[1:-1, -2]),
                         v_bc_right_wall(vh3[1:, -1]))  # this won't change anything for u2
            f.left_wall(uh3, vh3, u_bc_left_wall(t + (a31 + a32) * dt), v_bc_left_wall(t + (a31 + a32) * dt))

            press_stage_3 = np.zeros([nx + 2, ny + 2])
            print('        pressure projection stage{} = True'.format(2))
            u3, v3, press_stage_3, iter3 = f.ImQ_bcs(uh3, vh3, Coef, pn, p_bcs)
            iteration_i_3 = iter3
            print('        iterations stage 2 = ', iter3)
            # apply bcs
            f.top_wall(u3, v3, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(u3, v3, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(u3, v3, u_bc_right_wall(u3[1:-1, -1]),
                         v_bc_right_wall(v3[1:, -2]))  # this won't change anything for u2
            f.left_wall(u3, v3, u_bc_left_wall(t + (a31 + a32) * dt), v_bc_left_wall(t + (a31 + a32) * dt))

            div3 = np.linalg.norm(f.div(u3, v3).ravel())
            print('        divergence of u3 = ', div3)
            urhs3 = f.urhs_bcs(u3, v3)
            vrhs3 = f.vrhs_bcs(u3, v3)

            uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2) + dt * b3 * (urhs3)
            vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2) + dt * b3 * (vrhs3)

            f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]),
                         v_bc_right_wall(vhnp1[1:, -1]))  # this won't change anything for u2
            f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

            # unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn,p_bcs)
            unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn, p_bcs)

            iteration_np1 += iter2
            print('iter_np1=', iter2)

            # apply bcs
            f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]),
                         v_bc_right_wall(vnp1[1:, -2]))  # this won't change anything for unp1
            f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        elif count > 2:
            # switch to two step Adam Bashforth
            print("Adam-Bashforth 2 Steps")
            print("----------------------")
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()

            un = u.copy()
            vn = v.copy()

            unm1 = usol[-2].copy()
            vnm1 = vsol[-2].copy()

            unm2 = usol[-3].copy()
            vnm2 = vsol[-3].copy()

            urhsn = f.urhs_bcs(un, vn)
            vrhsn = f.vrhs_bcs(un, vn)

            urhsnm1 = f.urhs_bcs(unm1, vnm1)
            vrhsnm1 = f.vrhs_bcs(unm1, vnm1)

            urhsnm2 = f.urhs_bcs(unm2, vnm2)
            vrhsnm2 = f.vrhs_bcs(unm2, vnm2)

            b1 = 23.0/12
            b2 = -16.0/12
            b3 = 5.0/12
            # Adam Bashforth two step method
            uhnp1 = un + dt * ( b1 * (urhsn) + b2 * (urhsnm1) + b3 * (urhsnm2))
            vhnp1 = vn + dt * ( b1 * (vrhsn) + b2 * (vrhsnm1) + b3 * (vrhsnm2))

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
            mp_field = np.zeros_like(unp1)
            mp_field[1:-1, 1] = m_p(t + dt, yu[:, 0])

            # apply bcs on urhs_bcs and vrhs_bcs
            # -----------------------------------
            # intermediate velocity used to apply bcs
            unp1_inter = unp1 + dt * f.urhs_bcs(unp1, vnp1)
            vnp1_inter = vnp1 + dt * f.vrhs_bcs(unp1, vnp1)
            # apply bcs to the intermediate velocity
            f.top_wall(unp1_inter, vnp1_inter, u_bc_top_wall, v_bc_top_wall)
            f.bottom_wall(unp1_inter, vnp1_inter, u_bc_bottom_wall, v_bc_bottom_wall)
            f.right_wall(unp1_inter, vnp1_inter, u_bc_right_wall(unp1_inter[1:-1, -2]),
                         v_bc_right_wall(vnp1_inter[1:, -1]))
            f.left_wall(unp1_inter, vnp1_inter, u_bc_left_wall(t + 2 * dt), v_bc_left_wall(t + 2 * dt))
            # now only use the u and v rhs
            unp1_rhs = (unp1_inter - unp1) / dt
            vnp1_rhs = (vnp1_inter - vnp1) / dt
            rhs_pp = np.zeros_like(pn)
            rhs_pp = f.div(unp1_rhs, vnp1_rhs) - mp_field

            ml = pyamg.ruge_stuben_solver(Coef)
            ptmp = ml.solve(rhs_pp[1:-1, 1:-1], tol=1e-12)

            nx = probDescription.nx
            ny = probDescription.ny
            post_press = np.zeros([ny + 2, nx + 2])
            post_press[1:-1, 1:-1] = ptmp.reshape([ny, nx])
            p_bcs(post_press)

        new_press = 11*press/6 -7*pn/6 +pnm1/3 #(third order)

        iteration_np1+=iter2
        t+=dt
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

        # maxbound = max(gradpx[1:-1, 1:].ravel())
        maxbound = max(unp1[1:-1, 1:].ravel())
        # minbound = min(gradpx[1:-1, 1:].ravel())
        minbound = min(unp1[1:-1, 1:].ravel())
        # plot of the pressure gradient in order to make sure the solution is correct
        # if count%100 == 0:
        #     print("a({})=".format(t),at(t))
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
# probDescription = ProbDescription(N=[128,16],L=[4,1],μ =0.01,dt = 0.01)
# AB3_channel_flow_unsteady(steps=500, return_stability=False, name='heun',alpha=0.9,theta=None,post_projection=False)