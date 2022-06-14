import numpy as np
from core.functions import func
import time
from core import singleton_classes as sc


def RK4_taylor_vortex_parametric_approx (steps = 3,return_stability=False,name='regular',guess=None,gamma={
    "22":0,"23":0,"33":0},project=[1,1,1],alpha=0.99,post_projection=False):
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
        print('timestep:{}'.format(count+1))
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

            d2,d3,d4 = project
            # define the pressure derivatives
            Pn = 11 / 6 * pn - 7 / 6 * pnm1 + pnm2 / 3
            Pn_p = (2 * pn - 3 * pnm1 + pnm2) / dt
            Pn_pp = (pn - 2 * pnm1 + pnm2) / dt / dt

            c3 = a31 + a32
            c4 = a41 + a42 + a43
            # define parameters

            # dependent on the parameters
            gamma_22 = gamma["22"]  # a21/2
            gamma_23 = gamma["23"]  # a21**2/6
            gamma_33 = gamma["33"]  # c3**2/6
            # ---------------------------------

            gamma_32 = (1 / 24 - gamma_22 * (a21 * a32 * b3 + a21 * a42 * b4)) / (a43 * b4 * c3)
            gamma_42 = -(a21 * b2 * gamma_22) / (b4 * c4) - (b3 * gamma_32 * c3) / (b4 * c4) + 1 / (6 * b4 * c4)
            gamma_43 = -(gamma_23 * (a21 * b2)) / (b4 * c4) - (gamma_33 * (b3 * c3)) / (b4 * c4) + 1 / (24 * b4 * c4)

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
            u2, v2, _, iter1 = f.ImQ(uh2, vh2, Coef, pn)
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            if guess == "third":
                p_approx = Pn + gamma_22 * dt * Pn_p + gamma_23 * dt*dt *Pn_pp
            else:
                p_approx = 0

            u2 = uh2 - a21 * dt * f.Gpx(p_approx)
            v2 = vh2 - a21 * dt * f.Gpy(p_approx)

        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)

        ## stage 3
        print('    Stage 3:')
        print('    --------')
        urhs2 = f.urhs(u2, v2)
        vrhs2 = f.vrhs(u2, v2)

        uh3 = u + dt * (a31 * urhs1  + a32 * urhs2 )
        vh3 = v + dt * (a31 * vrhs1  + a32 * vrhs2 )

        if d3 == 1:
            print('        pressure projection stage{} = True'.format(3))
            u3, v3, _, iter2 = f.ImQ(uh3, vh3, Coef, pn)
            print('        iterations stage 3 = ', iter2)

        elif d3 == 0:
            if guess == "third":
                p_approx =  Pn + gamma_32 * dt * Pn_p + gamma_33 * dt*dt *Pn_pp
            else:
                p_approx = 0

            u3 = uh3 - (a31 + a32) * dt * f.Gpx(p_approx)
            v3 = vh3 - (a31 + a32) * dt * f.Gpy(p_approx)

        div3 = np.linalg.norm(f.div(u3, v3).ravel())
        print('        divergence of u3 = ', div3)

        ## stage 4
        print('    Stage 4:')
        print('    --------')
        urhs3 = f.urhs(u3, v3)
        vrhs3 = f.vrhs(u3, v3)

        uh4 = u + dt * (a41 * urhs1 + a42 * urhs2 + a43 * urhs3  )
        vh4 = v + dt * (a41 * vrhs1 + a42 * vrhs2 + a43 * vrhs3  )

        if d4 == 1:
            print('        pressure projection stage{} = True'.format(4))
            u4, v4, _, iter4 = f.ImQ(uh4, vh4, Coef, pn)
            print('        iterations stage 4 = ', iter4)

        elif d4 == 0:
            if guess == "third":
                p_approx =  Pn + gamma_42 * dt * Pn_p + gamma_43 * dt*dt *Pn_pp
            else:
                p_approx = 0

            u4 = uh4 - (a41 + a42 + a43) * dt * f.Gpx(p_approx)
            v4 = vh4 - (a41 + a42 + a43) * dt * f.Gpy(p_approx)

        div4 = np.linalg.norm(f.div(u4, v4).ravel())
        print('        divergence of u4 = ', div4)

        uhnp1 = u + dt*b1*(urhs1)  + dt*b2*(urhs2) + dt*b3*(urhs3) +  dt*b4*(f.urhs(u4,v4))
        vhnp1 = v + dt*b1*(vrhs1)  + dt*b2*(vrhs2) + dt*b3*(vrhs3) +  dt*b4*(f.vrhs(u4,v4))

        unp1,vnp1,press,iter3= f.ImQ(uhnp1,vhnp1,Coef,pn)
        time_end = time.time()

        if post_projection:
            # post processing projection
            uhnp1_star = u + dt * (f.urhs(unp1, vnp1))
            vhnp1_star = v + dt * (f.vrhs(unp1, vnp1))

            _, _, post_press, _ = f.ImQ(uhnp1_star, vhnp1_star, Coef, pn)

        new_press = 25*press/12 -23*pn/12 +13*pnm1/12 - pnm2/4

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