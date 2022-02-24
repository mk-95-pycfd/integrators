import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc


def RK76_taylor_vortex (steps = 3,return_stability=False,name='regular',guess=None,project=[1,1,1,1,1,1,1],alpha=0.99,post_projection=False):
    # problem description
    probDescription = sc.ProbDescription()
    f = func(probDescription,'periodic')
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()
    # define exact solutions
    a = 2*np.pi
    b = 2*np.pi
    uexact = lambda a, b, x, y, t: 1 - np.cos(a*(x - t))*np.sin(b*(y - t))*np.exp(-(a**2 + b**2)*μ*t)
    vexact = lambda a, b, x, y, t: 1 + np.sin(a*(x - t))*np.cos(b*(y - t))*np.exp(-(a**2 + b**2)*μ*t)

    dpdxexact = lambda x, t: -np.pi * np.exp(-16 * np.pi ** 2 * μ * t) * np.sin(np.pi * (4 * t - 4 * x))

    integ_dpdx_exact = lambda x, t1, t2: 1 / dt * (np.pi * (
            np.cos(4 * np.pi * t1 - 4 * np.pi * x) / (1024 * np.pi ** 5 * μ ** 4 * np.exp(16 * np.pi ** 2 * t1 * μ) \
                                                      + 64 * np.pi ** 3 * μ ** 2 * np.exp(
                16 * np.pi ** 2 * t1 * μ)) + np.sin(4 * np.pi * t1 - 4 * np.pi * x) / (
                    256 * np.pi ** 4 * μ ** 3 * np.exp(16 * np.pi ** 2 * t1 * μ) \
                    + 16 * np.pi ** 2 * μ * np.exp(16 * np.pi ** 2 * t1 * μ)) - np.exp(
        -16 * np.pi ** 2 * t1 * μ) * np.sin(4 * np.pi * t1 - 4 * np.pi * x) / (16 * np.pi ** 2 * μ) \
            - np.exp(-16 * np.pi ** 2 * t1 * μ) * np.cos(4 * np.pi * t1 - 4 * np.pi * x) / (
                    64 * np.pi ** 3 * μ ** 2)) - np.pi * (np.cos(4 * np.pi * t2 - 4 * np.pi * x) / (
            1024 * np.pi ** 5 * μ ** 4 * np.exp(16 * np.pi ** 2 * t2 * μ) \
            + 64 * np.pi ** 3 * μ ** 2 * np.exp(16 * np.pi ** 2 * t2 * μ)) + np.sin(
        4 * np.pi * t2 - 4 * np.pi * x) / (256 * np.pi ** 4 * μ ** 3 * np.exp(
        16 * np.pi ** 2 * t2 * μ) + 16 * np.pi ** 2 * μ * np.exp(16 * np.pi ** 2 * t2 * μ)) \
                                                          - np.exp(-16 * np.pi ** 2 * t2 * μ) * np.sin(
                4 * np.pi * t2 - 4 * np.pi * x) / (16 * np.pi ** 2 * μ) - np.exp(-16 * np.pi ** 2 * t2 * μ) * np.cos(
                4 * np.pi * t2 - 4 * np.pi * x) / (64 * np.pi ** 3 * μ ** 2)))

    #     # define some boiler plate
    t = 0.0
    tend = steps
    end_time = steps * dt
    count = 0
    print('dt=',dt)

    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.zeros([ny+2, nx + 2]) # include ghost cells
    u0[1:-1,1:] = uexact(a,b,xu,yu,0) # initialize the interior of u0
    # same thing for the y-velocity component
    v0 = np.zeros([ny +2, nx+2]) # include ghost cells
    v0[1:,1:-1] = vexact(a,b,xv,yv,0)
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

    iterations =[]

    Coef = f.A()

    is_stable =True
    stability_counter =0
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
    target_ke = ken_exact - alpha*(ken_exact-final_KE)
    print('time = ',t)
    print('ken_new = ',ken_new)
    print('ken_exc = ',ken_exact)
    while count < tend:
        RK76 = sc.RK76(name)
        a21 = RK76.a21
        a31 = RK76.a31
        a32 = RK76.a32
        a41 = RK76.a41
        a42 = RK76.a42
        a43 = RK76.a43
        a51 = RK76.a51
        a52 = RK76.a52
        a53 = RK76.a53
        a54 = RK76.a54
        a61 = RK76.a61
        a62 = RK76.a62
        a63 = RK76.a63
        a64 = RK76.a64
        a65 = RK76.a65
        a71 = RK76.a71
        a72 = RK76.a72
        a73 = RK76.a73
        a74 = RK76.a74
        a75 = RK76.a75
        a76 = RK76.a76
        b1  = RK76.b1
        b2  = RK76.b2
        b3  = RK76.b3
        b4  = RK76.b4
        b5  = RK76.b5
        b6  = RK76.b6
        b7  = RK76.b7
        print('timestep:{}'.format(count+1))
        print('-----------')
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 =  np.zeros_like(u)
        pnm2 =  np.zeros_like(u)
        pnm3 = np.zeros_like(u)
        pnm4 = np.zeros_like(u)
        pnm5 = np.zeros_like(u)
        pnm6 = np.zeros_like(u)

        f1x = np.zeros_like(pn)
        f1y = np.zeros_like(pn)
        f2x = np.zeros_like(pn)
        f2y = np.zeros_like(pn)
        f3x = np.zeros_like(pn)
        f3y = np.zeros_like(pn)
        if count >7:
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            pnm2 = psol[-3].copy()
            pnm3 = psol[-4].copy()
            pnm4 = psol[-5].copy()
            pnm5 = psol[-6].copy()
            pnm6 = psol[-7].copy()
            d2,d3,d4,d5,d6,d7 = project

        elif count <= 7: # compute pressures for 2 time steps
            d2 = 1
            d3 = 1
            d4 = 1
            d5 = 1
            d6 = 1
            d7 = 1

        ## stage 1
        print('    Stage 1:')
        print('    --------')
        time_start = time.time()
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
        uh2 = u + a21 * dt * urhs1
        vh2 = v + a21 * dt * vrhs1

        if d2 == 1:
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, p2, iter1 = f.ImQ(uh2, vh2, Coef, pn)
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            if guess == "third":
                p_approx = ((49 *a21)/20) *pn + (-((71 * a21)/20))*pnm1 +((79 *a21)/20)*pnm2 + (-((163 *a21)/60))*pnm3 + ((31 *a21)/30)*pnm4 + (-(a21/6))*pnm5
                # p_approx =  (11*a21/6 * pn - 7*a21/6 * pnm1 + a21/3 * pnm2)
            else:
                p_approx = 0
            u2 = uh2 - dt * f.Gpx(p_approx)
            v2 = vh2 - dt * f.Gpy(p_approx)
        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)

        ## stage 3
        print('    Stage 3:')
        print('    --------')
        urhs2 = f.urhs(u2, v2)
        vrhs2 = f.vrhs(u2, v2)

        uh3 = u + dt * (a31 * urhs1  + a32 * urhs2)
        vh3 = v + dt * (a31 * vrhs1  + a32 * vrhs2)

        if d3 == 1:
            print('        pressure projection stage{} = True'.format(3))
            u3, v3, p3, iter2 = f.ImQ(uh3, vh3, Coef, pn)
            print('        iterations stage 3 = ', iter2)

        elif d3 == 0:
            if guess == "third":
                c3 = a32 + a31
                p_approx =  (7/180 * (116 *a21 *a32 + 63 *c3)) * pn + (-(116/9) *a21 *a32 - (71 *c3)/20)*pnm1 +((589 *a21 *a32)/36 + (79 *c3)/20)*pnm2 \
                            + (-(427/36)*a21 *a32 - (163 *c3)/60) *pnm3 + ((167 *a21 *a32)/36 + (31 *c3)/30)*pnm4 +(-(137/180) *a21 *a32 - c3/6)*pnm5
                # p_approx =  ((11*c3/6 + 2 *a32*a21)*pn + (-7*(a32+a31)/6 - 3 * a32*a21) * pnm1 + (c3/3 + a32*a21)*pnm2)
            else:
                p_approx = 0

            u3 = uh3 - dt * f.Gpx(p_approx)
            v3 = vh3 - dt * f.Gpy(p_approx)
        div3 = np.linalg.norm(f.div(u3, v3).ravel())
        print('        divergence of u3 = ', div3)

        ## stage 4
        print('    Stage 4:')
        print('    --------')
        urhs3 = f.urhs(u3, v3)
        vrhs3 = f.vrhs(u3, v3)

        uh4 = u + dt * (a41 * urhs1 + a42 * urhs2 + a43 * urhs3 )
        vh4 = v + dt * (a41 * vrhs1 + a42 * vrhs2 + a43 * vrhs3 )

        if d4 == 1:
            print('        pressure projection stage{} = True'.format(4))
            u4, v4, p4, iter4 = f.ImQ(uh4, vh4, Coef, pn)
            print('        iterations stage 4 = ', iter4)

        elif d4 == 0:
            if guess == "third":
                c4 = a41 + a42 + a43
                sum1 = a42 * a21 + a43 * (a31 + a32)
                sum2 = a43 * a32 * a21
                p_approx =  ((11*c4/6 + 2 * sum1 + sum2)*pn + (-7*c4/6 -3*sum1-2*sum2)*pnm1 + (c4/3 + sum1 +sum2) * pnm2)
            else:
                p_approx = 0

            u4 = uh4 -dt * f.Gpx(p_approx)
            v4 = vh4 -dt * f.Gpy(p_approx)

        div4 = np.linalg.norm(f.div(u4, v4).ravel())
        print('        divergence of u4 = ', div4)

        ## stage 5
        print('    Stage 5:')
        print('    --------')
        urhs4 = f.urhs(u4, v4)
        vrhs4 = f.vrhs(u4, v4)

        uh5 = u + dt * (a51 * urhs1 + a52 * urhs2 + a53 * urhs3 + a54 * urhs4)
        vh5 = v + dt * (a51 * vrhs1 + a52 * vrhs2 + a53 * vrhs3 + a54 * vrhs4)

        if d5 == 1:
            print('        pressure projection stage{} = True'.format(5))
            u5, v5, p5, iter5 = f.ImQ(uh5, vh5, Coef, pn)
            print('        iterations stage 5 = ', iter5)

        elif d5 == 0:
            # todo: need to change these
            if guess == "third":
                c4 = a41 + a42 + a43
                sum1 = a42 * a21 + a43 * (a31 + a32)
                sum2 = a43 * a32 * a21
                p_approx = ((11 * c4 / 6 + 2 * sum1 + sum2) * pn + (-7 * c4 / 6 - 3 * sum1 - 2 * sum2) * pnm1 + (
                            c4 / 3 + sum1 + sum2) * pnm2)
            else:
                p_approx = 0

            u5 = uh5 - dt * f.Gpx(p_approx)
            v5 = vh5 - dt * f.Gpy(p_approx)

        div5 = np.linalg.norm(f.div(u5, v5).ravel())
        print('        divergence of u5 = ', div5)

        ## stage 6
        print('    Stage 6:')
        print('    --------')
        urhs5 = f.urhs(u5, v5)
        vrhs5 = f.vrhs(u5, v5)

        uh6 = u + dt * (a61 * urhs1 + a62 * urhs2 + a63 * urhs3 + a64 * urhs4 + a65 * urhs5)
        vh6 = v + dt * (a61 * vrhs1 + a62 * vrhs2 + a63 * vrhs3 + a64 * vrhs4 + a65 * vrhs5)

        if d6 == 1:
            print('        pressure projection stage{} = True'.format(6))
            u6, v6, p6, iter6 = f.ImQ(uh6, vh6, Coef, pn)
            print('        iterations stage 6 = ', iter6)

        elif d6 == 0:
            # todo: need to change these
            if guess == "third":
                c4 = a41 + a42 + a43
                sum1 = a42 * a21 + a43 * (a31 + a32)
                sum2 = a43 * a32 * a21
                p_approx = ((11 * c4 / 6 + 2 * sum1 + sum2) * pn + (-7 * c4 / 6 - 3 * sum1 - 2 * sum2) * pnm1 + (
                        c4 / 3 + sum1 + sum2) * pnm2)
            else:
                p_approx = 0

            u6 = uh6 - dt * f.Gpx(p_approx)
            v6 = vh6 - dt * f.Gpy(p_approx)

        div6 = np.linalg.norm(f.div(u6, v6).ravel())
        print('        divergence of u6 = ', div6)

        ## stage 7
        print('    Stage 7:')
        print('    --------')
        urhs6 = f.urhs(u6, v6)
        vrhs6 = f.vrhs(u6, v6)

        uh7 = u + dt * (a71 * urhs1 + a72 * urhs2 + a73 * urhs3 + a74 * urhs4 + a75 * urhs5 + a76 * urhs6)
        vh7 = v + dt * (a71 * vrhs1 + a72 * vrhs2 + a73 * vrhs3 + a74 * vrhs4 + a75 * vrhs5 + a76 * vrhs6)

        if d7 == 1:
            print('        pressure projection stage{} = True'.format(7))
            u7, v7, p7, iter7 = f.ImQ(uh7, vh7, Coef, pn)
            print('        iterations stage 7 = ', iter7)

        elif d7 == 0:
            # todo: need to change these
            if guess == "third":
                c4 = a41 + a42 + a43
                sum1 = a42 * a21 + a43 * (a31 + a32)
                sum2 = a43 * a32 * a21
                p_approx = ((11 * c4 / 6 + 2 * sum1 + sum2) * pn + (-7 * c4 / 6 - 3 * sum1 - 2 * sum2) * pnm1 + (
                        c4 / 3 + sum1 + sum2) * pnm2)
            else:
                p_approx = 0

            u7 = uh7 - dt * f.Gpx(p_approx)
            v7 = vh7 - dt * f.Gpy(p_approx)

        div7 = np.linalg.norm(f.div(u7, v7).ravel())
        print('        divergence of u7 = ', div7)

        uhnp1 = u + dt*b1*(urhs1)  + dt*b2*(urhs2) + dt*b3*(urhs3) +  dt*b4*(urhs4) +  dt*b5*(urhs5) +  dt*b6*(urhs6) +  dt*b7*(f.urhs(u7,v7))
        vhnp1 = v + dt*b1*(vrhs1)  + dt*b2*(vrhs2) + dt*b3*(vrhs3) +  dt*b4*(vrhs4) +  dt*b5*(vrhs5) +  dt*b6*(vrhs6) +  dt*b7*(f.vrhs(u7,v7))

        unp1,vnp1,press,iter3= f.ImQ(uhnp1,vhnp1,Coef,pn)
        time_end = time.time()

        if post_projection:
            # post processing projection
            uhnp1_star = u + dt * (f.urhs(unp1, vnp1))
            vhnp1_star = v + dt * (f.vrhs(unp1, vnp1))

            _, _, post_press, _ = f.ImQ(uhnp1_star, vhnp1_star, Coef, pn)

        new_press = (49/20) * press + (-71/20) * pn + (79/20) * pnm1 + (-163/60) * pnm2 + (31/30) * pnm3 + (-1/6) * pnm4

        psol.append(press)
        cpu_time = time_end - time_start
        print('cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1,vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('Mass residual:',residual)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)

        iterations.append(iter1+iter2+iter3)
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
            if stability_counter >5:
                print('not stable !!!!!!!!')
                break
            else:
                stability_counter+=1
        else:
            is_stable = True
            print('is_stable = ',is_stable)
            if ken_new<target_ke and count > 30:
                break
        ken_old = ken_new.copy()

        #plot of the pressure gradient in order to make sure the solution is correct
        # if count %10 == 0:
        #     # # plt.contourf(usol[-1][1:-1,1:])
        #     plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        #     plt.colorbar()
        #     plt.show()
        count+=1
    # error between the exact pressure gradient average and the pseudo-pressure
    # ---------------------------------------------------------------------------
    gradpx = lambda phi: ((phi[1:-1, 1:] - phi[1:-1, :-1]) / dx)

    average_exact_dpdx = integ_dpdx_exact(xu, end_time - dt, end_time)
    average_dpdx = gradpx(press)

    diff = np.linalg.norm(average_exact_dpdx.ravel() - average_dpdx.ravel(), np.inf)
    # plt.semilogy(xu[2,:],np.abs(average_exact_dpdx - average_dpdx)[2,:])
    # plt.show()
    print('        error={}'.format(diff))
    if return_stability:
        return is_stable

    else:
        return diff, [div_n,div2,div3,div_np1], is_stable, unp1[1:-1,1:].ravel()