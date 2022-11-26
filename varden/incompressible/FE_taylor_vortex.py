import numpy as np
from core.functions import func
from core.variable_density_functions import vardenfunc, SpatialOperators, BoundaryConditions
import time
from core import singleton_classes as sc


def FE_Taylor_Vortex_scalar (steps=3, return_stability=False,alpha=0.99):
    # problem description
    probDescription = sc.ProbDescription()
    SpatialOperators.set_problemDescription(probDescription)

    # Defining aliases
    bcs = BoundaryConditions
    sops = SpatialOperators
    fvc = sops.Calculus

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
    bcs.periodic(u0)
    bcs.periodic(v0)

    # define viscosity
    mu = np.ones([ny + 2, nx + 2]) * μ  # include ghost cells
    # define density
    density = np.ones([ny + 2, nx + 2])
    # initialize pressure
    p0 = np.zeros_like(u0)

    # initialize the scalar
    phi0 = np.zeros([ny+2,nx+2]); # include ghost cells
    A = 1
    sigx = 0.25
    sigy = 0.25
    phi0[1:-1,1:-1] = A * np.exp(-(xcc-lx/2)**2/2/sigx**2 -(ycc-ly/2)**2/2/sigy**2)

    bcs.periodic(phi0)

    # define the matrix
    Coef = f.A(density)

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
        dt = probDescription.get_dt()
        print('timestep:{}; time= {:4f}  dt={:7f}'.format(count+1,t,dt))
        print('------------------------------------------')
        p_n = psol[-1].copy()
        u_n = usol[-1].copy()
        v_n = vsol[-1].copy()
        phi_n = phisol[-1].copy()
        # phi_np1 = phi_n +  dt * f.scalar_rhs(diff_coef,np.ones_like(u0), np.ones_like(v0),phi_n)
        phi_np1 = phi_n +  dt * f.scalar_rhs(0.0,np.ones_like(u0), np.zeros_like(v0),phi_n)
        bcs.periodic(phi_np1)

        uh = u_n + dt * f.xMomPartialRHS(u_n, v_n,mu,density)
        vh = v_n + dt * f.yMomPartialRHS(u_n, v_n,mu,density)

        bcs.periodic(uh)
        bcs.periodic(vh)

        # update the coef matrix for the poisson equation
        Coef = f.A(density)
        # pressure equation
        p = f.Psolve(uh,vh,ci=1,MatCoef=Coef,atol=1e-10)
        bcs.periodic(p)

        u_np1 = uh - dt * fvc.GradXScalar(p)
        v_np1 = vh - dt * fvc.GradYScalar(p)
        bcs.periodic(u_np1)
        bcs.periodic(v_np1)

        div_np1 = np.linalg.norm(fvc.div_vect(u_np1, v_np1).ravel())
        # div_np1 = np.linalg.norm(f.div_vect(u0, v0).ravel())
        print("div = {}".format(div_np1))
        phisol.append(phi_np1)
        psol.append(p)
        usol.append(u_np1)
        vsol.append(v_np1)

        count += 1
        t += dt

        f.timeStepSelector(u_np1,v_np1,viscosity=mu,density=density,inner_scale=0.5,outer_scale=0.5)

        #plot of the pressure gradient in order to make sure the solution is correct
        if count%100 ==0:
            # plt.contourf(usol[-1][sops.FieldSlice.P])
            plt.contourf(phisol[-1][sops.FieldSlice.P])
            # plt.contourf(f.div_vect(u_np1,v_np1))
            # plt.contourf(sops.View(fvc.GradXScalar(p),"P"))
            plt.colorbar()
            plt.show()

    if return_stability:
        return True
    else:
        return False, [], True, u_np1[1:-1, 1:-1].ravel()

import matplotlib.pyplot as plt
probDescription = sc.ProbDescription(N=[32,32],L=[1,1],μ =1e-1,dt = 1e-10)
FE_Taylor_Vortex_scalar (steps = 2000)