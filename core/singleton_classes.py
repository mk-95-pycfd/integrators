import numpy as np


class SingletonDecorator:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        if self.instance == None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


@SingletonDecorator
class ProbDescription:
    def __init__(self, N, L, μ, dt):
        #     # define some boiler plate
        self.nx, self.ny = N
        self.lx, self.ly = L
        self.dx, self.dy = [l / n for l, n in zip(L, N)]
        self.μ = μ
        self.dt = dt

        self.dt_post_processing = None
        self.coef = None

        self.cell_centerted_coordinates()
        self.x_staggered_coordinates()
        self.y_staggered_coordinates()

    def cell_centerted_coordinates(self):
        # cell centered coordinates
        xx = np.linspace(self.dx / 2.0, self.lx - self.dx / 2.0, self.nx, endpoint=True)
        yy = np.linspace(self.dy / 2.0, self.ly - self.dy / 2.0, self.ny, endpoint=True)
        self.xcc, self.ycc = np.meshgrid(xx, yy)

    def x_staggered_coordinates(self):
        # x-staggered coordinates
        yy = np.linspace(self.dy / 2.0, self.ly - self.dy / 2.0, self.ny, endpoint=True)
        xxs = np.linspace(0, self.lx, self.nx + 1, endpoint=True)
        self.xu, self.yu = np.meshgrid(xxs, yy)

    def y_staggered_coordinates(self):
        # y-staggered coordinates
        xx = np.linspace(self.dx / 2.0, self.lx - self.dx / 2.0, self.nx, endpoint=True)
        yys = np.linspace(0, self.ly, self.ny + 1, endpoint=True)
        self.xv, self.yv = np.meshgrid(xx, yys)

    def get_gridPoints(self):
        return (self.nx, self.ny)

    def get_differential_elements(self):
        return (self.dx, self.dy)

    def get_domain_length(self):
        return (self.lx, self.ly)

    def get_cell_centered(self):
        return (self.xcc, self.ycc)

    def get_XVol(self):
        return (self.xu, self.yu)

    def get_YVol(self):
        return (self.xv, self.yv)

    def get_dt(self):
        return self.dt

    def get_mu(self):
        return self.μ

    def set_mu(self,val):
        self.μ = val

    def set_dt(self, dt):
        self.dt = dt

    def set_resolution(self,other):
        self.nx,self.ny = other
        self.__grid_spacing()
        self.cell_centerted_coordinates()
        self.x_staggered_coordinates()
        self.y_staggered_coordinates()

    def set_domain_size(self,other):
        self.lx,self.ly = other
        self.__grid_spacing()
        self.cell_centerted_coordinates()
        self.x_staggered_coordinates()
        self.y_staggered_coordinates()

    def __grid_spacing(self):
        self.dx, self.dy = [l / n for l, n in zip([self.lx,self.ly], [self.nx,self.ny])]

    def get_dt_post_processing(self):
        return self.dt_post_processing
    def set_dt_post_processing(self,dt):
        self.dt_post_processing = dt

# @SingletonDecorator
class RK2:
    def __init__(self, name,theta=None):
        self.name = name
        self.theta = theta
        self.coefs()

    def coefs(self):
        if self.name == 'heun':
            self.a21 = 1.0
            self.b1 = 1.0 / 2.0
            self.b2 = 1.0 / 2.0

        elif self.name == 'midpoint':
            self.a21 = 1.0 / 2.0
            self.b1 = 0.0
            self.b2 = 1.0

        elif self.name=='theta' and self.theta!=None:
            self.a21 = self.theta
            self.b1 = 1.0 -1.0 / (self.theta*2.0)
            self.b2 = 1.0 / (self.theta*2.0)

@SingletonDecorator
class RK3:
    def __init__(self, name):
        self.name = name
        self.coefs()

    def coefs(self):
        if self.name == 'regular':
            self.a21 = 1.0 / 2
            self.a31 = -1
            self.a32 = 2
            self.b1 = 1.0 / 6
            self.b2 = 2.0 / 3
            self.b3 = 1.0 / 6

        elif self.name == 'heun':
            self.a21 = 1.0 / 3
            self.a31 = 0
            self.a32 = 2.0 / 3
            self.b1 = 1.0 / 4
            self.b2 = 0
            self.b3 = 3.0 / 4

        elif self.name == 'ralston':
            self.a21 = 1.0 / 2
            self.a31 = 0
            self.a32 = 3.0 / 4
            self.b1 = 2.0 / 9
            self.b2 = 1.0 / 3
            self.b3 = 4.0 / 9

        elif self.name == 'ssp':
            self.a21 = 1.0
            self.a31 = 1.0 / 4
            self.a32 = 1.0 / 4
            self.b1 = 1.0 / 6
            self.b2 = 1.0 / 6
            self.b3 = 2.0 / 3


@SingletonDecorator
class RK4:
    def __init__(self, name):
        self.name = name
        self.coefs()

    def coefs(self):
        if self.name == 'regular':
            self.a21 = 1.0 / 2.0
            self.a31 = 0
            self.a32 = 1.0 / 2.0
            self.a41 = 0
            self.a42 = 0
            self.a43 = 1
            self.b1 = 1.0 / 6.0
            self.b2 = 1.0 / 3.0
            self.b3 = 1.0 / 3.0
            self.b4 = 1.0 / 6.0

        elif self.name == '3/8':
            self.a21 = 1.0 / 3.0
            self.a31 = -1.0 / 3.0
            self.a32 = 1.0
            self.a41 = 1
            self.a42 = -1
            self.a43 = 1
            self.b1 = 1.0 / 8.0
            self.b2 = 3.0 / 8.0
            self.b3 = 3.0 / 8.0
            self.b4 = 1.0 / 8.0

        elif self.name == 'sanderse':
            self.a21 = 1.0
            self.a31 = 3.0 / 8.0
            self.a32 = 1.0 / 8.0
            self.a41 = -1.0 / 8.0
            self.a42 = -3.0 / 8.0
            self.a43 = 3.0 /2
            self.b1 = 1.0 / 6.0
            self.b2 = -1.0 / 18.0
            self.b3 = 2.0 / 3.0
            self.b4 = 2.0 / 9.0

@SingletonDecorator
class RK54:
    def __init__(self, beta):
        if beta ==0:
            raise Exception("beta has to be different that zero")
        self.beta = beta
        self.coefs()

    def coefs(self):
        self.a21 = 1/4
        self.a31 = 1/2
        self.a32 = 0
        self.a41 = 0
        self.a42 = 1/2
        self.a43 = 1/4
        self.a51 = 0
        self.a52 = 1/6/self.beta
        self.a53 = -1/3/self.beta
        self.a54 = 1/6/self.beta
        self.b1  = -self.beta
        self.b2  = 2/3
        self.b3  = -1/3
        self.b4  = 2/3
        self.b5  = self.beta


@SingletonDecorator
class RK76:
    def __init__(self, name):
        self.name = name
        self.coefs()

    def coefs(self):
        if self.name == 'regular':
            self.a21 = 1/3
            self.a31 = 0
            self.a32 = 2/3
            self.a41 = 1/12
            self.a42 = 1/3
            self.a43 = -1/12
            self.a51 = 25/48
            self.a52 = -55/24
            self.a53 = 35/48
            self.a54 = 15/8
            self.a61 = 3/20
            self.a62 = -11/24 # this was the mistake it was -11/20
            self.a63 = -1/8
            self.a64 = 1/2
            self.a65 = 1/10
            self.a71 = -261/260
            self.a72 = 33/13
            self.a73 = 43/156
            self.a74 = -118/39
            self.a75 = 32/195
            self.a76 = 80/39
            self.b1  = 13/200
            self.b2  = 0
            self.b3  = 11/40
            self.b4  = 11/40
            self.b5  = 4/25
            self.b6  = 4/25
            self.b7  = 13/200

@SingletonDecorator
class DIRK2:
    def __init__(self, name, theta=None):
        self.name = name
        self.theta=theta
        self.coefs()

    def coefs(self):
        if self.name == 'midpoint':
            # this is a first order method
            self.a11 = 1.0 / 2.0
            self.a21 = -1.0 / 2.0
            self.a22 =  2.0
            self.b1 = -1.0 / 2.0
            self.b2 = 3.0 / 2.0

        elif self.name == 'qz':
            self.a11 = 1.0 / 4.0
            self.a21 = 1.0 / 2.0
            self.a22 = 1.0 / 4.0
            self.b1 = 1.0 / 2.0
            self.b2 = 1.0 / 2.0

        elif self.name =='pr':
            self.a11 = self.theta
            self.a21 = 1.0 - 2.0* self.theta
            self.a22 = self.theta
            self.b1 = 1.0 / 2.0
            self.b2 = 1.0 / 2.0

        elif self.name =='Crouzeix':
            # two stages third order accurate
            self.a11 = 1.0/2.0 + np.sqrt(3.0)/6.0
            self.a21 = -np.sqrt(3.0)/3.0
            self.a22 = 1.0/2.0 + np.sqrt(3.0)/6.0
            self.b1 = 1.0 / 2.0
            self.b2 = 1.0 / 2.0

class DIRK3:
    def __init__(self, name, theta=None):
        self.name = name
        self.theta = theta
        self.coefs()

    def coefs(self):
        if self.name =='3stages':
            # three stages third order accurate
            x = 0.4358665215
            self.a11 = x
            self.a21 = (1-x)/2
            self.a22 = x
            self.a31 = -3*x**2/2 + 4*x -1/4
            self.a32 = 3*x**2/2 -5*x + 5/4
            self.a33 = x
            self.b1 = -3*x**2/2 + 4*x -1/4
            self.b2 = 3*x**2/2 -5*x + 5/4
            self.b3 = x

            # print("b1= {}, b2= {}, b3= {}".format(self.b1,self.b2,self.b3))

        elif self.name =='3stages4':
            # three stages third order accurate
            # x = 1.06858
            # x = 0.30254
            x = 0.12889
            self.a11 = x
            self.a21 = 1/2-x
            self.a22 = x
            self.a31 = 2*x
            self.a32 = 1-4*x
            self.a33 = x
            self.b1 = 1/6/(1-2*x)**2
            self.b2 = (3*(1-2*x)**2-1)/(3*(1-2*x)**2)
            self.b3 = 1/6/(1-2*x)**2

            # print("b1= {}, b2= {}, b3= {}".format(self.b1,self.b2,self.b3))

        elif self.name =='Crouzeix34':
            x = 2*np.cos(np.pi/18)/np.sqrt(3)
            self.a11 = (1+x)/2
            self.a21 = -x/2
            self.a22 = (1+x)/2
            self.a31 = 1+x
            self.a32 = -(1+2*x)
            self.a33 = (1+x)/2
            self.b1 = 1/6/x**2
            self.b2 = 1-1/(3*x**2)
            self.b3 = 1/6/x**2
