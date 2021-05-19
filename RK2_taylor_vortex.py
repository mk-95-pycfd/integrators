import numpy as np
from functions import func
import time
import statistics
import singleton_classes as sc
import scipy
import matplotlib.pyplot as plt

def RK2_taylor_vortex(steps=3, return_stability=False, name='heun', guess=None, project=[1],alpha=0.9,theta=None):
    # problem description
    probDescription = sc.ProbDescription()
    f = func(probDescription,'periodic')
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

    pexact=lambda x,y,t: (-8 * np.sin(np.pi * t) ** 4 * np.sin(np.pi * y) ** 4 - 2 * np.sin(np.pi * t) ** 4 - 2 * np.sin(np.pi * y) ** 4 - 5 * np.cos(
        2 * np.pi * t) / 2 + 5 * np.cos(4 * np.pi * t) / 8 - 5 * np.cos(2 * np.pi * y) / 2 + 5 * np.cos(4 * np.pi * y) / 8 - np.cos(
        np.pi * (2 * t - 4 * y)) / 4 + np.cos(np.pi * (2 * t - 2 * y)) + np.cos(np.pi * (2 * t + 2 * y)) - np.cos(
        np.pi * (2 * t + 4 * y)) / 4 - 3 * np.cos(np.pi * (4 * t - 4 * y)) / 16 - np.cos(np.pi * (4 * t - 2 * y)) / 4 - np.cos(
        np.pi * (4 * t + 2 * y)) / 4 + np.cos(np.pi * (4 * t + 4 * y)) / 16 + 27 / 8) * np.exp(-16 * np.pi ** 2 * μ * t) - np.exp(
        -16 * np.pi ** 2 * μ * t) * np.cos(np.pi * (-4 * t + 4 * x)) / 4

    dpdxexact = lambda x,t:-np.pi*np.exp(-16*np.pi**2*μ*t)*np.sin(np.pi*(4*t - 4*x))
    #     # define some boiler plate
    t = 0.0
    tend = steps
    count = 0
    print('dt=', dt)

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

    # declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1 = np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol = []
    usol.append(u0)

    vsol = []
    vsol.append(v0)

    psol = []
    psol.append(p0)

    iterations = []

    Coef = f.A()

    is_stable = True

    # # u and v num cell centered
    ucc = 0.5 * (u0[1:-1, 2:] + u0[1:-1, 1:-1])
    vcc = 0.5 * (v0[2:, 1:-1] + v0[1:-1, 1:-1])

    uexc = uexact(a, b, xu, yu, t)
    vexc = vexact(a, b, xv, yv, t)
    # u and v exact cell centered
    uexc_cc = 0.5 * (uexc[:, :-1] + uexc[:, 1:])
    vexc_cc = 0.5 * (vexc[:-1, :] + vexc[1:, :])

    # compute of kinetic energy
    ken_new = np.sum(ucc.ravel() ** 2 + vcc.ravel() ** 2) / 2
    ken_exact = np.sum(uexc_cc.ravel() ** 2 + vexc_cc.ravel() ** 2) / 2
    ken_old = ken_new
    final_KE = nx * ny
    target_ke = ken_exact - alpha * (ken_exact - final_KE)
    print('time = ', t)
    print('ken_new = ', ken_new)
    print('ken_exc = ', ken_exact)
    stability_counter = 0
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
        # pnm2 = np.zeros_like(u)
        if count > 1: # change to 2 if high order pressure is needed
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            # pnm2 = psol[-3].copy()# only needed for high order pressure
            f1x, f1y = f.Guess([pn, pnm1], order=guess, integ='RK2', type=name,theta=theta)
            d2, = project

        elif count <= 1:  # compute pressures for 2 time steps # change to 2 if high order pressure is needed
            d2 = 1
            f1x, f1y = f.Guess([pn, pnm1], order=None, integ='RK2', type=name,theta=theta)
            iteration_i_2 = 0
            iteration_np1 = 0
        ## stage 1

        print('    Stage 1:')
        print('    --------')
        time_start = time.clock()
        u1 = u.copy()
        v1 = v.copy()

        # Au1
        urhs1 = f.urhs(u1, v1)
        vrhs1 = f.vrhs(u1, v1)

        # divergence of u1
        div_n = np.linalg.norm(f.div(u1, v1).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh2 = u + a21 * dt * (urhs1 - f1x)
        vh2 = v + a21 * dt * (vrhs1 - f1y)

        if d2 == 1:
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, press_stage_2, iter1 = f.ImQ(uh2, vh2, Coef, pn)
            # u2, v2, press_stage_2, iter1 = f.ImQ(uh2, vh2, Coef, (3*pn-pnm1)/2,tol=1e-10)
            iteration_i_2 +=iter1
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            u2 = uh2
            v2 = vh2
        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)
        urhs2 = f.urhs(u2, v2)
        vrhs2 = f.vrhs(u2, v2)

        uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2)
        vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2)

        unp1, vnp1, press, iter2 = f.ImQ(uhnp1, vhnp1, Coef, pn,tol=1e-10)
        # unp1, vnp1, press, iter2 = f.ImQ(uhnp1, vhnp1, Coef, (3*pn-pnm1)/2,atol=1e-6,tol=1e-16) # midpoint
        # unp1, vnp1, press, iter2 = f.ImQ(uhnp1, vhnp1, Coef, press_stage_2)

        # new_press =  4*pn -9*pnm1/ 2 +3 * pnm2 / 2 #second order (working)

        iteration_np1+=iter2

        # # post processing projection
        # unp1r = dt * f.urhs(unp1, vnp1)
        # vnp1r = dt * f.vrhs(unp1, vnp1)
        #
        # _, _, press, _ = f.ImQ_post_processing(unp1r, vnp1r, Coef, press)
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

        iterations.append(iter1 + iter2)
        # # u and v num cell centered
        ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])

        uexc = uexact(a, b, xu, yu, t)
        vexc = vexact(a, b, xv, yv, t)
        # u and v exact cell centered
        uexc_cc = 0.5 * (uexc[:, :-1] + uexc[:, 1:])
        vexc_cc = 0.5 * (vexc[:-1, :] + vexc[1:, :])
        t += dt

        # compute of kinetic energy
        ken_new = np.sum(ucc.ravel() ** 2 + vcc.ravel() ** 2) / 2
        ken_exact = np.sum(uexc_cc.ravel() ** 2 + vexc_cc.ravel() ** 2) / 2
        print('time = ', t)
        print('ken_new = ', ken_new)
        print('target_ken=', target_ke)
        print('ken_exc = ', ken_exact)
        print('(ken_new - ken_old)/ken_old = ', (ken_new - ken_old) / ken_old)
        if (((ken_new - ken_old) / ken_old) > 0 and count > 1) or np.isnan(ken_new):
            is_stable = False
            print('is_stable = ', is_stable)
            if stability_counter > 5:
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

        # plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        gradpx = (psol[-1][1:-1, 1:] - psol[-1][1:-1, :-1]) / dx

        maxbound = max(gradpx[1:-1, 1:].ravel())
        minbound = min(gradpx[1:-1, 1:].ravel())
        # plot of the pressure gradient in order to make sure the solution is correct
        # if count%10 == 0:
        #     plt.imshow(gradpx[1:-1,1:],origin='bottom',cmap='jet',vmax=maxbound, vmin=minbound)
        #     # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        #     v = np.linspace(minbound, maxbound, 4, endpoint=True)
        #     plt.colorbar(ticks=v)
        #     plt.title('time: {}'.format(t))
        #     plt.show()
        count += 1
    diff = np.linalg.norm(uexact(a, b, xu, yu, t).ravel() - unp1[1:-1, 1:].ravel(), np.inf)
    print('        error={}'.format(diff))
    if return_stability:
        return is_stable
    else:
        return diff, [iteration_i_2, iteration_np1], is_stable, unp1[1:-1, 1:].ravel()
