import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc


def FE_taylor_vortex (steps=3, return_stability=False,alpha=0.99):
    # problem description
    probDescription = sc.ProbDescription()
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
    p0 = np.zeros([nx+2,ny+2]); # include ghost cells

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
    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        ## stage 1
        pn = np.zeros_like(u0)
        if count > 1:
            pn = psol[-1].copy()

        print('    Stage 1:')
        print('    --------')
        time_start = time.clock()
        u = usol[-1].copy()
        v = vsol[-1].copy()

        # divergence of u1
        div_n = np.linalg.norm(f.div(u, v).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh = u +  dt * f.urhs(u, v)
        vh = v +  dt * f.vrhs(u, v)
        unp1, vnp1, press, iter = f.ImQ(uh, vh, Coef, pn)
        total_iteration+=iter
        time_end = time.clock()
        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:',residual)
        print('iterations:',iter)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        iterations.append(iter)
        # # u and v num cell centered
        ucc = 0.5*(u[1:-1,2:] + u[1:-1,1:-1])
        vcc = 0.5*(v[2:,1:-1] + v[1:-1,1:-1])

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
        #plot of the pressure gradient in order to make sure the solution is correct
        # plt.contourf(usol[-1][1:-1,1:])
        # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # plt.colorbar()
        # plt.show()
        count+=1
    diff = np.linalg.norm(uexact(a,b,xu,yu,t).ravel()-unp1[1:-1,1:] .ravel(),np.inf)
    print('        error={}'.format(diff))
    if return_stability:
        return is_stable
    else:
        return diff, [total_iteration], is_stable, unp1[1:-1, 1:].ravel()

# import matplotlib.pyplot as plt
# from singleton_classes import ProbDescription
#
# probDescription = ProbDescription(N=[32,32],L=[1,1],μ =0.01,dt = 0.001)
# FE_taylor_vortex (steps = 2000)