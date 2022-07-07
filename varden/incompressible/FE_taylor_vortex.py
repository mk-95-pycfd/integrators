import numpy as np
from core.functions import func
from core.variable_density_functions import vardenfunc
import time
from core import singleton_classes as sc


def FE_Taylor_Vortex_scalar (steps=3, return_stability=False,alpha=0.99):
    # problem description
    probDescription = sc.ProbDescription()
    f = vardenfunc(probDescription, 'periodic')
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    diff_coef = 0.1
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()
    lx, ly = probDescription.get_domain_length()
    #     # define some boiler plate
    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)

    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # define exact solutions
    a = 2 * np.pi
    b = 2 * np.pi
    uf = 1
    vf = 1
    uexact = lambda a, b, x, y, t: uf - np.cos(a * (x - uf * t)) * np.sin(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)
    vexact = lambda a, b, x, y, t: vf + np.sin(a * (x - uf * t)) * np.cos(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    v0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    u0[1:-1, 1:] = uexact(a, b, xu, yu, 0)
    v0[1:, 1:-1] = vexact(a, b, xv, yv, 0)
    f.periodic_u(u0)
    f.periodic_v(v0)

    # define viscosity
    mu = np.ones([ny + 2, nx + 2]) * μ  # include ghost cells

    # initialize pressure
    p0 = np.zeros_like(u0)

    # initialize the scalar
    phi0 = np.zeros([ny+2,nx+2]); # include ghost cells
    A = 1
    sigx = 0.25
    sigy = 0.25
    phi0[1:-1,1:-1] = A * np.exp(-(xcc-lx/2)**2/2/sigx**2 -(ycc-ly/2)**2/2/sigy**2)

    f.periodic_scalar(phi0)

    # define the matrix
    Coef = f.A()

    # create storage
    phisol = []
    phisol.append(phi0)
    usol = []
    usol.append(u0)
    vsol = []
    vsol.append(v0)
    psol = []
    psol.append(p0)
    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        p_n = psol[-1].copy()
        u_n = usol[-1].copy()
        v_n = vsol[-1].copy()
        phi_n = phisol[-1].copy()
        phi_np1 = phi_n +  dt * f.scalar_rhs(diff_coef,u0, v0,phi_n)
        f.periodic_scalar(phi_np1)

        uh = u_n + dt * f.xMomPartialRHS(u_n, v_n,mu)
        vh = v_n + dt * f.yMomPartialRHS(u_n, v_n,mu)

        f.periodic_u(uh)
        f.periodic_v(vh)

        # pressure equation
        p = f.Psolve(uh,vh,ci=1,MatCoef=Coef,atol=1e-10)
        f.periodic_scalar(p)

        u_np1 = uh - dt * f.GradXScalar(p)
        v_np1 = vh - dt * f.GradYScalar(p)
        f.periodic_u(u_np1)
        f.periodic_v(v_np1)

        # u_np1, v_np1, press, _ = f.ImQ(uh, vh, Coef, p_n)

        div_np1 = np.linalg.norm(f.div_vect(u_np1, v_np1).ravel())
        # div_np1 = np.linalg.norm(f.div_vect(u0, v0).ravel())
        print("div = {}".format(div_np1))
        phisol.append(phi_np1)
        psol.append(p)
        usol.append(u_np1)
        vsol.append(v_np1)

        count += 1
        t += dt

        #plot of the pressure gradient in order to make sure the solution is correct
        if count%1 ==0:
            # plt.contourf(vsol[-1][1:-1,1:])
            # plt.contourf(f.div_vect(u_np1,v_np1))
            plt.contourf(f.View(f.GradXScalar(p),"P"))
            plt.colorbar()
            plt.show()

    if return_stability:
        return True
    else:
        return False, [], True, u_np1[1:-1, 1:-1].ravel()

import matplotlib.pyplot as plt

probDescription = sc.ProbDescription(N=[32,32],L=[1,1],μ =0.01,dt = 0.001)
FE_Taylor_Vortex_scalar (steps = 2000)