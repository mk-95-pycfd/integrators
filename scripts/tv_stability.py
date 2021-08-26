from core.singleton_classes import ProbDescription
import numpy as np
import os

from taylor_vortex.RK2_taylor_vortex import RK2_taylor_vortex
from taylor_vortex.RK3_taylor_vortex import RK3_taylor_vortex


# taylor vortex
#---------------
probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)
tend = 100000
U = 1.42
n = 15
# Res = [i for i in np.logspace(-1,2,n)]
Res = [0.5]
def run_fname():
    nsteps = int(tend/probDescription.get_dt())
    # here add your scheme:
    #------------------------
    is_stable = RK2_taylor_vortex(steps=nsteps,return_stability=True, name='heun', guess=None, project=[1],alpha=0.999)
    # is_stable = RK3_taylor_vortex(steps=nsteps,return_stability=True, name='regular', guess=None, project=[1, 1],alpha=0.999)

    print('is_stable = {}'.format(is_stable))
    return is_stable



def samesign(a,b):
    return a*b>=0

def bisect_CFL(CFLs,mu,old_CFL_low=0,old_is_stable_low=0,tol=1e-2):
    CFL_L = min(CFLs)
    CFL_H = max(CFLs)
    dx, dy = probDescription.get_differential_elements()
    Re = U*dx/mu
    print('-----------------------------------------------------------------')
    print('-----CFL_L={}----CFL_H={}-----Re={}-----'.format(CFL_L,CFL_H,Re))
    print('-----------------------------------------------------------------')



    dt_low = CFL_L*dx/U
    dt_high = CFL_H*dx/U

    diff = CFL_H-CFL_L
    midpoint = (CFL_L+CFL_H)/2

    probDescription.set_mu(mu)
    probDescription.set_dt(dt_low)
    is_stable_low = run_fname() if old_CFL_low!=CFL_L else old_is_stable_low
    probDescription.set_dt(dt_high)
    is_stable_high = run_fname()

    old_CFL_low = CFL_L
    old_is_stable_low = is_stable_low
    if samesign(is_stable_low,is_stable_high):
        if is_stable_high:
            CFL_L += diff
            CFL_H += diff
        else:
            if (diff<CFL_L and not(is_stable_low)):
                CFL_H -= diff
                CFL_L -= diff
            else:
                CFL_H = midpoint
    else:
        CFL_H = midpoint

    err =  abs(CFL_H-CFL_L)/CFL_L

    if err <tol and (is_stable_low!=is_stable_high) or err<1e-4:
        print('-----------------------------------------------------------------')
        print('-----CFL={}----Error={}-----'.format(midpoint,err))
        print('-----------------------------------------------------------------')
        return midpoint, err,is_stable_low,is_stable_high
    else:
        return bisect_CFL([CFL_L,CFL_H],mu,old_CFL_low,old_is_stable_low)

integ = "RK21-"
for Re in Res:
    CFL_max,CFL_min = (4,0.5)
    dx,dy = probDescription.get_differential_elements()
    mu = U*dx/Re
    probDescription.set_mu(mu)
    CFL,err,stable_low, stable_high = bisect_CFL([CFL_max,CFL_min],mu)
    with open("{}-stability.txt".format(integ), "a") as myfile:
        myfile.write("{},{},{},{},{}".format(CFL,Re,err,stable_low, stable_high))
        myfile.write("\n")
    print(CFL, Re, err, stable_low, stable_high)