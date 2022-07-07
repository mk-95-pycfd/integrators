import numpy as np
from core.functions import func
from core.variable_density_functions import vardenfunc
import time
from core import singleton_classes as sc


def RK3_addvection_diffusion (steps=3, name="heun", return_stability=False, theta=None):
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

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.ones([ny + 2, nx + 2])  # include ghost cells
    v0 = np.ones([ny + 2, nx + 2])  # include ghost cells
    f.periodic_u(u0)
    f.periodic_v(v0)

    # initialize the pressure
    phi0 = np.zeros([ny+2,nx+2]); # include ghost cells
    A = 1
    sigx = 0.25
    sigy = 0.25
    phi0[1:-1,1:-1] = A * np.exp(-(xcc-lx/2)**2/2/sigx**2 -(ycc-ly/2)**2/2/sigy**2)

    f.periodic_scalar(phi0)

    phisol = []
    phisol.append(phi0)

    RK3 = sc.RK3(name)
    a21 = RK3.a21
    a31 = RK3.a31
    a32 = RK3.a32
    b1 = RK3.b1
    b2 = RK3.b2
    b3 = RK3.b3

    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        phi_n = phisol[-1]

        # stage 2
        phi_2 = phi_n + a21 * dt * f.scalar_rhs(diff_coef,u0,v0,phi_n)

        f.periodic_scalar(phi_2)

        # stage 3
        phi_3 = phi_n + a31 * dt * f.scalar_rhs(diff_coef, u0, v0, phi_n) \
                + a32 * dt * f.scalar_rhs(diff_coef,u0, v0,phi_2)

        f.periodic_scalar(phi_3)
        # t = t_{n+1}
        phi_np1 = phi_n +  b1 * dt * f.scalar_rhs(diff_coef,u0, v0,phi_n) \
                  +  b2 * dt * f.scalar_rhs(diff_coef,u0, v0,phi_2) \
                  +  b3 * dt * f.scalar_rhs(diff_coef,u0, v0,phi_3)

        f.periodic_scalar(phi_np1)

        phisol.append(phi_np1)

        count += 1
        t += dt

        #plot of the pressure gradient in order to make sure the solution is correct
        # if count%100 ==0:
        #     plt.contourf(phisol[-1][1:-1,1:])
        #     # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        #     plt.colorbar()
        #     plt.show()

    if return_stability:
        return True
    else:
        return False, [], True, phi_np1[1:-1, 1:-1].ravel()

# import matplotlib.pyplot as plt
#
# probDescription = sc.ProbDescription(N=[32,32],L=[1,1],μ =0.01,dt = 0.001)
# RK2_addvection_diffusion (steps = 2000,name="heun")