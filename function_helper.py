import numpy as np

class TestFunction():
    def f(self, x):
        raise NotImplementedError
    def grad(self, x):
        raise NotImplementedError
    def hess(self, x):
        raise NotImplementedError
    def third_order_tensor(self, x):
        raise NotImplementedError
    def evaluate(self, x, order=0):

        f_val = self.f(x)
        if order == 0:
            return f_val

        grad_val = self.grad(x)

        return f_val, grad_val
    
class M_x_sig(TestFunction):
    def __init__(self, sig, x0, fx, gradx, hessx):
        self.sig = sig
        self.x0 = np.atleast_1d(x0)
        self.fx = float(fx)
        self.gradx = np.atleast_1d(gradx)
        self.hessx = np.atleast_2d(hessx)
        self.n = np.shape(x0)[0]

    def f(self, x):
        return self.fx + np.dot(self.gradx,x-self.x0) + 1/2 * np.dot(self.hessx @ (x - self.x0),x - self.x0) + self.sig/2*np.linalg.norm(x - self.x0)**3

    def grad(self, x):
        return self.gradx + self.hessx @ (x - self.x0) + 3*self.sig/2 * np.linalg.norm(x - self.x0) * (x - self.x0)
    
class Omega_x_sig(TestFunction):
    def __init__(self, sig, x0, fx, gradx, hessx,thirdx):
        self.sig = sig
        self.x0 = np.atleast_1d(x0)
        self.fx = float(fx)
        self.gradx = np.atleast_1d(gradx)
        self.hessx = np.atleast_2d(hessx)
        self.thirdx = np.atleast_3d(thirdx)
        self.n = np.shape(x0)[0]

    def f(self, x):
        return self.fx + np.dot(self.gradx,x-self.x0) + 1/2 * np.dot(self.hessx @ (x - self.x0),x - self.x0) + (1/6) * np.einsum('ijk,i,j,k->', self.thirdx, x - self.x0, x - self.x0, x - self.x0) + self.sig/2*np.linalg.norm(x - self.x0)**4/4

    def grad(self, x):
        return self.gradx + self.hessx @ (x - self.x0) + 1/2 * np.einsum('ijk,j,k->i', self.thirdx, x - self.x0, x - self.x0) + self.sig/2 * np.linalg.norm(x - self.x0)**2 * (x - self.x0)


class f1(TestFunction):
    def f(self, x):
        return np.sin(x)

    def grad(self, x):
        return np.cos(x)
    
    def hess(self, x):
        return -np.sin(x)
    
    def third_order_tensor(self, x):
        return -np.cos(x)
    
class h1(TestFunction):
    def f(self, x):
        return (
            6*x**5
            -15*x**4
            +10*x**3
        )
    def grad(self, x):
        return (
            30*x**4
            -60*x**3
            +30*x**2
        )
    
    def hess(self, x):
        return (
        120*x**3
        -180*x**2
        +60*x
        )   
    
    def third_order_tensor(self, x):
        return (
        360*x**2
        -360*x
        +60
    )
 
class h2(TestFunction):
    def f(self, x):
        y = h1().f((x)%1)+x//1
        return np.abs(y)
    
    def grad(self, x):
        y = h1().grad((x)%1)
        return (np.where(x > 0, y, -y))
    
    def hess(self, x):
        y = h1().hess((x)%1)
        return (np.where(x > 0, y, -y))
    
    def third_order_tensor(self, x):
        y = h1().third_order_tensor((x)%1)
        return (np.where(x > 0, y, -y))
    
class h3(TestFunction):
    def f(self, x):
        return (
        -20*x**7
        +70*x**6
        -84*x**5
        +35*x**4
    )

    def grad(self, x):
        return (
        -140*x**6
        +420*x**5
        -420*x**4
        +140*x**3
    )
    
    def hess(self, x):
        return (
        -840*x**5
        +2100*x**4
        -1680*x**3
        +420*x**2
    )
    
    def third_order_tensor(self, x):
        return (
        -4200*x**4
        +8400*x**3
        -5040*x**2
        +840*x
    )

class f4(TestFunction):
    def f(self, x):
        y = h3().f((x)%1)+x//1
        return np.abs(y)
    
    def grad(self, x):
        y = h3().grad((x)%1)
        return (np.where(x > 0, y, -y))
    
    def hess(self, x):
        y = h3().hess((x)%1)
        return (np.where(x > 0, y, -y))
    
    def third_order_tensor(self, x):
        y = h3().third_order_tensor((x)%1)
        return (np.where(x > 0, y, -y))
    
class polytest(TestFunction):
    def f(self, x):
        return (
            6*x**4
            -15*x**3
            +10*x**2
        )
    def grad(self, x):
        return (
            24*x**3
            -45*x**2
            +20*x**1
        )
    
    def hess(self, x):
        return (
        72*x**2
        -90*x*
        +20
        )   
    
    def third_order_tensor(self, x):
        return (
        144*x
        -90
        
    )

    
class rosenbrock(TestFunction):

    def __init__(self, d=2, a=1.0, b=100.0):
        self.d = d
        self.a = a
        self.b = b

    def f(self, x):
        val = 0.0

        for i in range(self.d-1):
            val+= self.b*(x[i]**2 - x[i+1])**2 + (x[i] - self.a)**2

        return val
    
    def grad(self, x):

        grd = np.zeros((self.d))

        grd[0] = 4*self.b*(x[0]**3 - x[0]*x[1]) + 2*(x[0] - self.a)

        for i in range(1,self.d-1):
            grd[i] = 4*self.b*(x[i]**3 - x[i]*x[i+1]) + 2*(x[i] - self.a) + 2*self.b*(x[i] - x[i-1]**2)

        grd[self.d-1] = 2*self.b*(x[self.d-1] - x[self.d-2]**2)

        return grd 
    
    def hess(self, x):
        

        hss = np.zeros((self.d,self.d))

        hss[0][0] = 4*self.b*(3*x[0]**2-x[1]) + 2.0
        hss[0][1] = -4*self.b*x[0]

        for i in range(1,self.d-1):
            hss[i][i-1] = -4*self.b*x[i-1]
            hss[i][i] = 4*self.b*(3*x[i]**2-x[i+1]) + 2.0 + 2*self.b
            hss[i][i+1] = -4*self.b*x[i]

        hss[self.d-1][self.d-2] = -4*self.b*x[self.d-2]
        hss[self.d-1][self.d-1] = 2*self.b

        return hss
    
    def third_order_tensor(self, x):

        third = np.zeros((self.d,self.d,self.d))

        third[0][0][0] = 24*self.b*x[0]
        third[0][0][1] = -4*self.b
        third[0][1][0] = -4*self.b


        for i in range(1,self.d-1):
            third[i][i-1][i-1] = -4*self.b
            third[i][i][i] = 24*self.b*x[i]
            third[i][i][i+1] = -4*self.b
            third[i][i+1][i] = -4*self.b

        third[self.d-1][self.d-2][self.d-2] = -4*self.b

        return third
