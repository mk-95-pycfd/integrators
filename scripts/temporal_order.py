import numpy as np
import json
from core.singleton_classes import ProbDescription, RK4

from taylor_vortex.AB2_taylor_vortex import AB2_taylor_vortex

# from channel_flow_steady.FE_channel_flow_steady import FE_channel_flow
# from channel_flow_steady.RK2_channel_flow_steady import RK2_channel_flow
# from channel_flow_steady.RK3_channel_flow_steady import RK3_channel_flow
# from channel_flow_steady.RK4_channel_flow_steady import RK4_channel_flow
# from channel_flow_steady.AB2_channel_flow_steady import AB2_channel_flow
# from channel_flow_unsteady.FE_channel_flow_unsteady import FE_channel_flow_unsteady
# # from channel_flow_unsteady.RK2_channel_flow_unsteady import RK2_channel_flow_unsteady
# from channel_flow_unsteady.generalized_approx.RK2_channel_flow_unsteady import RK2_channel_flow_unsteady
# # from channel_flow_unsteady.RK3_channel_flow_unsteady import RK3_channel_flow_unsteady
# from channel_flow_unsteady.generalized_approx.RK3_channel_flow_unsteady import RK3_channel_flow_unsteady
# # from channel_flow_unsteady.RK4_channel_flow_unsteady import RK4_channel_flow_unsteady
# from channel_flow_unsteady.generalized_approx.RK4_channel_flow_unsteady import RK4_channel_flow_unsteady
# from channel_flow_unsteady.AB2_channel_flow_unsteady import AB2_channel_flow_unsteady
# from channel_flow_unsteady.AB3_channel_flow_unsteady import AB3_channel_flow_unsteady
# from taylor_vortex.RK3_taylor_vortex import RK3_taylor_vortex
# # from taylor_vortex.generalized_approx.RK3_taylor_vortex import RK3_taylor_vortex
# # from taylor_vortex.RK2_taylor_vortex import RK2_taylor_vortex
# from taylor_vortex.generalized_approx.RK2_taylor_vortex import RK2_taylor_vortex
# # from taylor_vortex.RK4_taylor_vortex import RK4_taylor_vortex
from taylor_vortex.generalized_approx.RK4_taylor_vortex import RK4_taylor_vortex
# from taylor_vortex.generalized_approx.RK4_taylor_vortex_parametric import RK4_taylor_vortex_parametric
from taylor_vortex.generalized_approx.RK3_taylor_vortex_parametric_approx import RK3_taylor_vortex_parametric_approx
from channel_flow_unsteady.generalized_approx.RK3_channel_flow_unsteady_parametric_approx import RK3_channel_flow_unsteady_parametric_approx
from taylor_vortex.DIRK2_taylor_vortex import DIRK2_taylor_vortex
# from taylor_vortex.generalized_approx.DIRK3_taylor_vortex import DIRK3_taylor_vortex
from taylor_vortex.generalized_approx.RK4_taylor_vortex_parametric_approx import RK4_taylor_vortex_parametric_approx
from channel_flow_unsteady.generalized_approx.RK4_channel_flow_unsteady_parametric_approx import RK4_channel_flow_unsteady_parametric_approx
from taylor_vortex.generalized_approx.RK76_taylor_vortex import RK76_taylor_vortex
from taylor_vortex.generalized_approx.RK54_taylor_vortex import RK54_taylor_vortex

# taylor vortex
# ---------------
# probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.025)

# channel flow steady
#--------------------
# ν = 0.1
# Uinlet = 1
# probDescription = ProbDescription(N=[4*16,16],L=[4,1],μ =ν,dt = 0.01)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)
# probDescription.set_dt(1e-4)

# channel flow unsteady_inlet
#-----------------------------
probDescription = ProbDescription(N=[4*16,16],L=[4,1],μ =0.1,dt = 0.0025)

levels = 4       # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]

timesteps = [10*rt**i for i in range(0,levels) ]

# to run multiple cases
#========================
# RK2 integrators
# -----------------
# stages_projections = [[0],[0],[1]]
# guesses = [None,'first',None]
# keys = ['RK20*','RK20','RK21']
# integrator_name = 'midpoint'
# theta = None

# # RK3 integrators
# #-----------------
# stages_projections = [[0,0],[0,0],[0,0],[0,1],[1,0],[1,1]]
# guesses = [None,'first','second','second','second',None]
# keys = ['RK300*','RK300**','RK300','RK301','RK310','RK311']
# integrator_name = 'heun'
# theta = None

# # RK4 integrators
# # #-----------------
# stages_projections = [[0,0,0],[0,0,0],[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
# guesses = [None,'second','third','third','third','third','third','third','third',None]
# keys = ['RK4000**','RK4000*','RK4000','RK4001','RK4010','RK4011','RK4100','RK4101','RK4110','RK4111']
# integrator_name = '3/8'
# theta = None

# to run single case
#========================
# FE integrator
# ----------------
# stages_projections = [[1]]
# guesses = [None]
# keys = ['FE']
# integrator_name = 'FE'
# theta = None

# RK2 integrator
# # ----------------
# stages_projections = [[1]]
# guesses = [None]
# keys = ['RK21']
# integrator_name = 'heun'
# theta = None

# RK3 integrator
# ----------------
stages_projections = [[1,0]]
guesses = ["second"]
keys = ['RK310']
integrator_name = 'regular'
theta = None
gamma={"22":0.25}

time_func = lambda t: (1.1- 1*np.exp(-t/2 ))# 2*(1-np.exp(-t))#np.sin((np.pi / 6) * np.sin(t / 2)) * np.cos(t / 2.0)#
time_func_prime = lambda t: 1#2*np.exp(-t)
time_func_dprime = lambda t: 0#-2*np.exp(-t)
user_bcs_time_func = [time_func,time_func_prime, time_func_dprime]

# # RK4 integrator
# #----------------
# stages_projections = [[0,0,0]]
# guesses = ["third"]
# keys = ['RK4000']
# # integrator_name = 'c2'
# # integrator_name = 'regular'
# integrator_name = '3/8'
# # integrator_name = 'sanderse3'
# # theta = None
# # RK4_integ = RK4(integrator_name)
# # RK4_integ.c2 = 12
#
# gamma = {"22": 0.25,"23":0.5,"33":0}
# time_func = lambda t:2*(1-np.exp(-t))
# time_func_prime = lambda t: 2*np.exp(-t)
# time_func_dprime = lambda t: -2*np.exp(-t)
# user_bcs_time_func = [time_func,time_func_prime]

# # RK54 integrator
# #----------------
# stages_projections = [[1,0,1,1]]
# guesses = ['third']
# keys = ['RK54']
# integrator_name = 'regular'
# theta = None

# # RK76 integrator
# #----------------
# stages_projections = [[1,0,1,1,1,1]]
# guesses = ['third']
# keys = ['RK76']
# integrator_name = 'regular'
# theta = None

# # DIRK2 integrator
# # ----------------
# stages_projections = [[1,1]]
# guesses = [None]
# keys = ['DIRK211']
# integrator_name = 'CN'
# theta = None

# DIRK3 integrator
# ----------------
# stages_projections = [[0,0,0]]
# guesses = ["second"]
# keys = ['DIRK3000']
# integrator_name = 'Crouzeix34'
# theta = None

# AB2 integrator
# ----------------
# stages_projections = [[1]]
# guesses = [None]
# keys = ['AB2']
# integrator_name = 'heun'
# theta = None

# AB2 integrator
# ----------------
# stages_projections = [[1]]
# guesses = [None]
# keys = ['AB3']
# integrator_name = 'heun'
# theta = None

# # RK3 integrator Capuano
# #-----------------------
# stages_projections = [[0, 0],[1, 0], [0, 1],[1,1]]
# guesses = ['capuano_ci_00','capuano_ci_10','capuano_ci_01',None]
# # guesses = ['capuano_00','capuano_10','capuano_01',None]
# keys = ['RK300','RK310','RK301','RK311']
# # keys = ['FC_ci_00','FC_ci_10','FC_ci_01']
# # keys = ['KS_ci_00','KS_ci_10','KS_ci_01']
# integrator_name = 'regular'
# theta = None

# file_name = 'channel_flow_2D_constructed_pressure_16x4_16x1_nu_0.1_{}.json'.format('AB3_heun')
file_name = 'taylor_vortex_32x32_nu_1e-3_{}.json'.format('RK76_regular')
# file_name = 'channel_flow_unsteady_64x16_nu_0.1_{}.json'.format('RK4001_ssp')

directory = './verification/temporal_accuracy/taylor_vortex/generalized_approx/RK76_taylor_vortex/u_np1/'

dict = {}
for proj, guess, key in zip(stages_projections,guesses,keys):
    phiAll = []
    for dt, nsteps in zip(dts, timesteps):
        probDescription.set_dt(dt)

        # taylor vortex
        #---------------
        # e, divs, _, phi =FE_taylor_vortex(steps = nsteps)
        # e, divs, _, phi = RK2_taylor_vortex(steps=nsteps, name=integrator_name, guess=guess, project=proj,theta=None,post_projection=False)
        # e, divs, _, phi =RK3_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj,post_projection=False)
        # e, divs, _, phi =RK3_taylor_vortex_parametric_approx(steps = nsteps,name=integrator_name,guess=guess,project=proj,alpha_2_approx=alpha_2_approx,post_projection=False)
        # e, divs, _, phi =RK4_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj,post_projection=False)
        # e, divs, _, phi =RK4_taylor_vortex_parametric(steps = nsteps,name=integrator_name,guess=guess,project=proj,post_projection=False)
        # e, divs, _, phi,_ =DIRK2_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj, alpha=0.99, theta= 0.5, Tol= 1e-8,post_projection=True)

        # e, divs, _, phi =RK54_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj,post_projection=False)
        # e, divs, _, phi =RK76_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj,post_projection=False)

        # e, divs, _, phi =AB2_taylor_vortex(steps = nsteps,name=integrator_name,post_projection=False)



        # channel flow
        #-------------
        # e, divs, _, phi =FE_channel_flow(steps = nsteps)
        # e, divs, _, phi = RK2_channel_flow(steps=nsteps, name=integrator_name, guess=guess, project=proj,theta=None,post_projection=False)
        # e, divs, _, phi = RK3_channel_flow(steps=nsteps, name=integrator_name, guess=guess, project=proj,post_projection=False)
        # e, divs, _, phi = RK4_channel_flow(steps=nsteps, name=integrator_name, guess=guess, project=proj,post_projection=False)
        # e, divs, _, phi =AB2_channel_flow(steps = nsteps,name=integrator_name,post_projection=False)

        # channel flow unsteady
        #----------------------
        # e, divs, _, phi = FE_channel_flow_unsteady(steps = nsteps)
        # e, divs, _, phi = RK2_channel_flow_unsteady(steps=nsteps, name=integrator_name, guess=guess, project=proj,theta=None,post_projection=False)
        # e, divs, _, phi = RK3_channel_flow_unsteady(steps=nsteps, name=integrator_name, guess=guess, project=proj,post_projection=False)
        # e, divs, _, phi = RK4_channel_flow_unsteady(steps=nsteps, name=integrator_name, guess=guess, project=proj,post_projection=True)
        # e, divs, _, phi =AB2_channel_flow_unsteady(steps = nsteps,name=integrator_name,post_projection=False)
        # e, divs, _, phi =AB3_channel_flow_unsteady(steps = nsteps,name=integrator_name,post_projection=False)

        # e, divs, _, phi =RK3_channel_flow_unsteady_parametric_approx(steps = nsteps,name=integrator_name,guess=guess,project=proj,alpha_2_approx=alpha_2_approx,post_projection=False)

        # Generalized approximations
        #============================
        # Taylor Vortex
        #---------------
        # e, divs, _, phi,_ = DIRK2_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj, alpha=0.99, Tol= 1e-10,post_projection=True)
        # e, divs, _, phi,_ = DIRK3_taylor_vortex(steps = nsteps,name=integrator_name,guess=guess,project=proj, alpha=0.99, Tol= 1e-10,post_projection=True)

        # e, divs, _, phi =RK3_taylor_vortex_parametric_approx(steps = nsteps,name=integrator_name,guess=guess,project=proj,gamma=gamma,post_projection=False)

        # e, divs, _, phi =RK4_taylor_vortex_parametric_approx(steps = nsteps,name=integrator_name,guess=guess,project=proj,gamma_43_approx=0.0,post_projection=False)


        # Generalized approximations
        #============================
        # Unsteady Channel Flow
        #-----------------------
        # e, divs, _, phi =RK4_channel_flow_unsteady_parametric_approx(steps = nsteps,name=integrator_name,
        #                                                              guess=guess,project=proj,gamma=gamma,
        #                                                              post_projection=False,
        #                                                              user_bcs_time_func=user_bcs_time_func)

        e, divs, _, phi =RK3_channel_flow_unsteady_parametric_approx(steps = nsteps,name=integrator_name,
                                                                     guess=guess,project=proj,gamma=gamma,
                                                                     post_projection=False,
                                                                     user_bcs_time_func=user_bcs_time_func)

        phiAll.append(phi)

    # local errors
    exact_err_mom = []
    errAll = []
    for i in range(0, levels - 1):
        diff = phiAll[i + 1] - phiAll[i]
        err = np.linalg.norm(diff, 2)
        print('error', err)
        errAll.append(err)

    # now compute order
    Order = []
    for i in range(0, levels - 2):
        Order.append(np.log(errAll[i + 1] / errAll[i]) / np.log(1.0 / rt))
        print('order: ', Order[-1])

    dict[key]={'dt':dts[:-1],"error": errAll}


import matplotlib.pyplot as plt
import matplotlib
fig_temp = plt.figure()
plt.loglog(np.array(dts)[:-1], errAll, 'o-', label='Error')
plt.loglog(np.array(dts),  0.5e5*np.array(dts), '--', label=r'$1^{rst}$')
plt.loglog(np.array(dts),  0.5e5*np.array(dts)**3, '--', label=r'$3^{rd}$')
plt.loglog(np.array(dts),  1e4*np.array(dts)**2, '--', label=r'$2^{nd}$')
# plt.loglog(np.array(dts),  0.5e4*np.array(dts)**4, '--', label=r'$4^{th}$')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$L_\infty$ Norm')
plt.legend()
# plt.savefig("instantaneous-press-np1-using-stage-pseudo-pressure.pdf",transparent=True)
plt.show()
#
# with open(directory+file_name,"w") as file:
#     json.dump(dict,file,indent=4)