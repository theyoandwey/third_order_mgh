import numpy as np

class TestFunction:
    def __init__(self, x0=None, m=None):
        if x0 is None:
            x0 = np.zeros((2))
        if m is None:
            m = 1
        self.x0 = x0
        self.m = m
        self.n = np.shape(x0)[0]

    def index2(self, n, j, k):
         return j * (j + 1) // 2 + k
    def index3(self, n, j, k,l):
         return j*(j+1)*(j+2)//6 + k*(k+1)//2 + l

    def f(self, x):
        return self.evaluate(x, order=0)

    def grad(self, x):
        _, g = self.evaluate(x, order=1)
        return g

    def hess(self, x):
        _, _, h = self.evaluate(x, order=2)
        return h

    def third_order_tensor(self, x):
        _, _, _, t = self.evaluate(x, order=3)
        return t
    def evaluate(self, x, order=3):

        f_val = self.f(x)
        if order == 0:
            return f_val

        grad_val = self.grad(x)
        if order == 1:
            return f_val, grad_val

        hess_val = self.hess(x)
        if order == 2:
            return f_val, grad_val, hess_val

        tensor_val = self.third_order_tensor(x)
        return f_val, grad_val, hess_val, tensor_val

class ChainRule(TestFunction):
    def __init__(self, x0=None, m=None):
        if x0 is None:
            x0 = np.array([0.0, 0.0, 0.0])
        if m is None:
            m = 3
        super().__init__(x0, m)

    def fi(self, x, i):
        return 0.0

    def gradi(self, x, i):
        return np.zeros(self.n)

    def hessi(self, x, i):
        return np.zeros((self.n, self.n))

    def third_order_tensori(self, x, i):
        return np.zeros((self.n, self.n, self.n))

    def collect(self, x, order=0):
        m, n = self.m, self.n
        indices = range(1, m + 1)
        fv = np.array([self.fi(x, i) for i in indices])
        if order == 0:
            return (fv,)
        G = np.array([self.gradi(x, i) for i in indices])
        if order == 1:
            return fv, G
        H = np.array([self.hessi(x, i) for i in indices])
        # copy lower triangle to upper (don't average)
        for k in range(m):
            H[k] = np.tril(H[k]) + np.tril(H[k], -1).T
        if order == 2:
            return fv, G, H
        T3 = np.array([self.third_order_tensori(x, i) for i in indices])
        # symmetrize from canonical entries (j >= k >= l)
        for idx in range(m):
            t = T3[idx]
            sym = np.zeros_like(t)
            for j in range(n):
                sym[j,j,j] = t[j,j,j]
                for k in range(j):
                    sym[j,j,k] = sym[j,k,j] = sym[k,j,j] = t[j,j,k]
                    sym[j,k,k] = sym[k,j,k] = sym[k,k,j] = t[j,k,k]
                    for l in range(k):
                        sym[j,k,l] = sym[j,l,k] = sym[k,j,l] = sym[k,l,j] = sym[l,j,k] = sym[l,k,j] = t[j,k,l]
            T3[idx] = sym
        return fv, G, H, T3

    # f = Î£ fiÂ²
    
    def evaluate(self, x, order=3):
        fv, G, H, T3 = self.collect(x, order=order) + (None,) * (4 - order - 1)

        f_val = np.dot(fv, fv)
        if order == 0:
            return f_val

        grad_val = 2.0 * G.T @ fv
        if order == 1:
            return f_val, grad_val

        GtG = G.T @ G
        hess_val = 2.0 * (np.einsum('i,ijk->jk', fv, H) + GtG)
        if order == 2:
            return f_val, grad_val, hess_val

        tensor_val = 2.0 * (
            np.einsum('i,ijkl->jkl', fv, T3) +
            np.einsum('ijk,il->jkl', H, G) +
            np.einsum('ijl,ik->jkl', H, G) +
            np.einsum('ikl,ij->jkl', H, G)
        )
        return f_val, grad_val, hess_val, tensor_val
#1--
class Rosenbrock(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([-1.2, 1.0])
        super().__init__(x0, 2)
    def f(self, x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    def grad(self, x):
        return np.array([400*x[0]*(x[0]**2-x[1])-2*(1-x[0]),200*(x[1]-x[0]**2)])
    def hess(self, x):
        hss = np.zeros([2,2])
        hss[0,0] = 1200*x[0]**2 - 400*x[1] + 2
        hss[1,0] = -400*x[0]
        hss[0,1] = hss[1,0]
        hss[1,1] = 200 
        return hss
    def third_order_tensor(self, x):
        trd = np.zeros([2,2,2])
        trd[0,0,0] = 2400*x[0]
        trd[1,0,0] = -400
        trd[0,1,0] = -400
        trd[0,0,1] = -400
        return trd

class FreudensteinAndRoth(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([4.0, 5.0])
        super().__init__(x0, 2)
    def f(self, x):
        val = 0.0
        x1 = x[0]
        x2 = x[1]

        val += (-13 + x1 + x2*((5 - x2)*x2 - 2))**2
        val += (-29 + x1 + x2*(x2*(x2+1) - 14))**2

        return val
    def grad(self, x):
        grd = np.zeros([2])
        x1 = x[0]
        x2 = x[1]

        grd[0] += 2*(-13 + x1 + x2*((5 - x2)*x2 - 2)) + 2*(-29 + x1 + x2*(x2*(x2+1) - 14))
        grd[1] += 2*(-13 + x1 + x2*((5 - x2)*x2 - 2))*(10*x2 - 3*x2**2 - 2) + 2*(-29 + x1 + x2*(x2*(x2+1) - 14))*(3*x2**2 +2*x2-14)

        return grd
    def hess(self, x):
        hss = np.zeros([2,2])
        x1 = x[0]
        x2 = x[1]
        hss[0,0] = 4
        hss[1,0] = 2*(10*x2 - 3*x2**2 - 2) + 2*(3*x2**2 + 2*x2 - 14)
        hss[0,1] = hss[1,0]
        hss[1,1] = 2*(10 - 6*x2)*(-13 + x1 + x2*((5 - x2)*x2 - 2)) + 2*(10*x2 - 3*x2**2 -2)*(10*x2 - 3*x2**2 - 2) + 2*(6*x2 + 2)*(-29 + x1 + x2*(x2*(x2+1) - 14)) + 2*(3*x2**2 + 2*x2 - 14)*(3*x2**2 + 2*x2 - 14)
        return hss
    def third_order_tensor(self, x):
        trd = np.zeros([2,2,2])
        x1 = x[0]
        x2 = x[1]
        trd[1,1,0] = 24.0
        trd[0,1,1] = trd[1,1,0]
        trd[1,0,1] = trd[1,1,0]
        trd[1,1,1] = -12*(-13 + x1 + x2*((5 - x2)*x2 - 2)) + 12*(-29 + x1 + x2*(x2*(x2 + 1) - 14)) + 6*(10 - 6*x2)*(10*x2 - 3*x2**2 - 2) + 6*(6*x2 + 2)*(3*x2**2 + 2*x2 - 14)
        return trd

class PowellBadlyScaled(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([0.0, 1.0])
        super().__init__(x0, 2)
    def f(self, x):
        val = 0.0
        x1 = x[0]
        x2 = x[1]

        val += (1e4*x1*x2 - 1)**2 + (np.exp(-x1) + np.exp(-x2) - 1.0001)**2

        return val
    def grad(self, x):
        grd = np.zeros([2])
        x1 = x[0]
        x2 = x[1]

        grd[0] += 2*(1e4*x1*x2 - 1)*1e4*x2 - 2*np.exp(-x1)*(np.exp(-x1) + np.exp(-x2) - 1.0001)
        grd[1] += 2*(1e4*x1*x2 - 1)*1e4*x1 - 2*np.exp(-x2)*(np.exp(-x1) + np.exp(-x2) - 1.0001)

        return grd
    def hess(self, x):
        hss = np.zeros([2,2])
        x1 = x[0]
        x2 = x[1]
        exp1 = np.exp(-x1)
        exp2 = np.exp(-x2)
        hss[0,0] = 2*1e8*x2**2 + 4*exp1**2 + 2*exp1*exp2 - 2.0002*exp1
        hss[1,0] = 4*1e8*x1*x2 + 2*exp1*exp2 - 2e4
        hss[0,1] = hss[1,0]
        hss[1,1] = 2*1e8*x1**2 + 4*exp2**2 + 2*exp1*exp2 - 2.0002*exp2
        return hss
    def third_order_tensor(self, x):
        trd = np.zeros([2,2,2])
        x1 = x[0]
        x2 = x[1]
        exp1 = np.exp(-x1)
        exp2 = np.exp(-x2)
        trd[0,0,0] = - 8*exp1**2 - 2*exp1*exp2 + 2.0002*exp1
        trd[1,0,0] = 4e8*x2 - 2*exp1*exp2
        trd[0,1,0] = trd[1,0,0]
        trd[0,0,1] = trd[1,0,0]
        trd[1,1,0] = 4e8*x1 - 2*exp1*exp2
        trd[1,0,1] = trd[1,1,0]
        trd[0,1,1] = trd[1,1,0]
        trd[1,1,1] = - 8*exp2**2 - 2*exp1*exp2 + 2.0002*exp2
        return trd
    
class BrownBadlyScaled(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([1.0, 1.0])
        super().__init__(x0, 3)
    def f(self, x):
        return (x[0]-1e6)**2 + (x[1]-2e-6)**2 + (x[0]*x[1]-2)**2
    def grad(self, x):
        grd = np.zeros([2])
        grd[0] = 2*(x[0]-1e6) + 2*x[1]*(x[0]*x[1]-2)
        grd[1] = 2*(x[1]-2e-6) + 2*x[0]*(x[0]*x[1]-2)
        return grd
    def hess(self, x):
        hss = np.zeros([2,2])
        x1 = x[0]
        x2 = x[1]
        hss[0,0] = 2 + 2*x2**2
        hss[0,1] = 4*x1*x2-4
        hss[1,0] = hss[0,1]
        hss[1,1] = 2 + 2*x1**2
        return hss
    
    def third_order_tensor(self, x):
        trd = np.zeros([2,2,2])
        x1 = x[0]
        x2 = x[1]
        trd[1,0,0] = 4*x2
        trd[1,1,0] = 4*x1
        trd[0,1,0] = trd[1,0,0]
        trd[0,0,1] = trd[1,0,0]
        trd[1,0,1] = trd[1,1,0]
        trd[0,1,1] = trd[1,1,0]
        return trd

class Beale(TestFunction) :

    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([1.0, 1.0])
        self.y = np.array([0.0, 1.5, 2.25, 2.625])
        super().__init__(x0, 3)


    def f(self, x):
        val = 0
        for i in range(1,self.m+1) : 
            val += (self.y[i] - x[0]*(1-x[1]**i))**2
        return val
    def grad(self, x):
        grd = np.zeros([2])
        for i in range(1,self.m+1) :
            grd[0] += -2*(1-x[1]**i)*(self.y[i]-x[0]*(1-x[1]**i))
            grd[1] += 2*i*x[0]*x[1]**(i-1)*(self.y[i]-x[0]*(1-x[1]**i))
        return grd
    def hess(self, x):
        hss = np.zeros([2,2])
        for i in range(1,self.m+1):
            hss[0,0] += 2*(1-x[1]**i)**2
            hss[0,1] += 4*i*x[0]*(x[1]**(2*i-1)-x[1]**(i-1))+2*i*self.y[i]*x[1]**(i-1)
            hss[1,1] += 2*i*((i-1)*x[0]*x[1]**(i-2)*self.y[i] + (2*i-1)*x[0]**2*x[1]**(2*i-2) - (i-1)*x[0]**2*x[1]**(i-2))
        hss[1,0] = hss[0,1]
        return hss
    def third_order_tensor(self, x):
        trd = np.zeros([2,2,2])
        for i in range(1,self.m+1) :
            trd[0,0,1] += -4*i*x[1]**(i-1)*(1-x[1]**i)
            trd[0,1,1] += 2*i*((i-1)*x[1]**(i-2)*self.y[i] + 2*(2*i-1)*x[0]*x[1]**(2*i-2) - 2*(i-1)*x[0]*x[1]**(i-2))
            trd[1,1,1] += 2*i*((i-1)*(i-2)*x[0]*x[1]**(i-3)*self.y[i] + (2*i-2)*(2*i-1)*x[0]**2*x[1]**(2*i-3) - (i-2)*(i-1)*x[0]**2*x[1]**(i-3))

        trd[0,1,0] = trd[0,0,1]
        trd[1,0,0] = trd[0,0,1]
        trd[1,0,1] = trd[0,1,1]
        trd[1,1,0] = trd[0,1,1]
        return trd

class JennrichAndSampson(TestFunction):
    def __init__(self, m=10, x0=None):
        if x0 is None:
            x0 = np.array([0.3, 0.4])
        super().__init__(x0, m)

    def fi(self, x, i):
        return 2 + 2*i - np.exp(i*x[0]) - np.exp(i*x[1])
    
    def dfij(self, x, i, j):
        return - i*np.exp(i*x[j])
    
    def d2fijj(self, x, i, j):
        return - i**2*np.exp(i*x[j])
    
    def d3fijjj(self, x, i, j):
        return - i**3*np.exp(i*x[j])

    def f(self, x):
        val = 0
        for i in range(self.m):
            val += self.fi(x, i+1)**2
        return val
    
    def grad(self, x):
        grd = np.zeros((self.n))
        for i in range(self.m):
            for j in range(self.n) :
                grd[j]+=2*self.fi(x, i+1)*self.dfij(x, i+1,j)
        return grd
    
    def hess(self, x):
        n = self.n
        m = self.m
        hss = np.zeros((n,n))
        dfi_vect = np.zeros((n))
        d2fi_vect = np.zeros((n))
        for i in range(m) :
            fi_val  = self.fi(x,i+1)
            for j in range(n):
                dfi_vect[j] = self.dfij(x,i+1,j)
                d2fi_vect[j] = self.d2fijj(x,i+1,j)
            for j in range(n):
                hss[j][j] += 2*(d2fi_vect[j]*fi_val  + dfi_vect[j]**2)
                for k in range(j):
                    hss[k][j] += 2*(dfi_vect[j]*dfi_vect[k])
        for j in range(n):
            for k in range(j):
                hss[j,k] = hss[k,j]
        return hss
    
    def third_order_tensor(self, x):
        n = self.n
        m = self.m
        trd = np.zeros((n,n,n))
        dfi_vect = np.zeros((n))
        d2fi_vect = np.zeros((n))
        d3fi_vect = np.zeros((n))
        for i in range(m) :
            fi_val  = self.fi(x,i+1)
            for j in range(n):
                dfi_vect[j] = self.dfij(x,i+1,j)
                d2fi_vect[j] = self.d2fijj(x,i+1,j)
                d3fi_vect[j] = self.d3fijjj(x,i+1,j)
            for j in range(n):
                trd[j][j][j] += 2*(d3fi_vect[j]*fi_val + 3*d2fi_vect[j]*dfi_vect[j])
                for k in range(j):
                    trd[j][j][k] += 2*(d2fi_vect[j]*dfi_vect[k])
                    trd[j][k][k] += 2*(d2fi_vect[k]*dfi_vect[j])
        for j in range(n):
            for k in range(j):
                trd[k][j][j] = trd[j][j][k]
                trd[j][k][j] = trd[j][j][k]
                trd[k][k][j] = trd[j][k][k]
                trd[k][j][k] = trd[j][k][k]
        return trd
    
class HelicalValley(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([-1.0, 0.0, 0.0])
        super().__init__(x0, 3)

    def theta(self, x):
        t = np.arctan2(x[1], x[0]) / (2 * np.pi)
        return t if t >= 0 else t + 1.0

    def fi(self, x, i):
        if i == 1:
            return 10*(x[2] - 10*self.theta(x))
        elif i == 2:
            return 10*(np.sqrt(x[0]**2 + x[1]**2) - 1)
        elif i == 3:
            return x[2]
        return 0.0

    def gradi(self, x, i):
        grd = np.zeros(self.n)
        r2 = x[0]**2 + x[1]**2
        if i == 1:
            grd[0] = 100/(2*np.pi) * x[1] / r2
            grd[1] = -100/(2*np.pi) * x[0] / r2
            grd[2] = 10.0
        elif i == 2:
            r = np.sqrt(r2)
            grd[0] = 10*x[0] / r
            grd[1] = 10*x[1] / r
        elif i == 3:
            grd[2] = 1.0
        return grd

    def hessi(self, x, i):
        hss = np.zeros((self.n, self.n))
        r2 = x[0]**2 + x[1]**2
        if i == 1:
            hss[0,0] = -100/np.pi * x[0]*x[1] / r2**2
            hss[1,1] =  100/np.pi * x[0]*x[1] / r2**2
            hss[1,0] =  100/(2*np.pi) * (x[0]**2 - x[1]**2) / r2**2
        elif i == 2:
            r3 = r2**(3/2)
            hss[0,0] =  10.0 * x[1]**2 / r3
            hss[1,1] =  10.0 * x[0]**2 / r3
            hss[1,0] = -10.0 * x[0]*x[1] / r3
        return hss

    def third_order_tensori(self, x, i):
        trd = np.zeros((self.n, self.n, self.n))
        r2 = x[0]**2 + x[1]**2
        if i == 1:
            r6 = r2**3
            trd[0,0,0] = -100/np.pi * x[1]*(x[1]**2 - 3*x[0]**2) / r6
            trd[1,1,1] =  100/np.pi * x[0]*(x[0]**2 - 3*x[1]**2) / r6
            trd[1,1,0] =  100/np.pi * x[1]*(x[1]**2 - 3*x[0]**2) / r6
            trd[1,0,0] = -100/np.pi * x[0]*(x[0]**2 - 3*x[1]**2) / r6
        elif i == 2:
            r5 = r2**(5/2)
            trd[0,0,0] = -30.0 * x[0] * x[1]**2 / r5
            trd[1,1,1] = -30.0 * x[1] * x[0]**2 / r5
            trd[1,1,0] =  10.0 * x[0] * (2*x[1]**2 - x[0]**2) / r5
            trd[1,0,0] =  10.0 * x[1] * (2*x[0]**2 - x[1]**2) / r5
        return trd

class Bard(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([1.0, 1.0, 1.0])
        self.y = [0.0, 0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]
        u = [i for i in range(16)]
        v = [16 - i for i in range(16)]
        w = [min(i, 16 - i) for i in range(16)]
        self.V = []
        for i in range(16):
            self.V.append([u[i], v[i], w[i]])
        super().__init__(x0, 15)

    def _vars(self, x, i):
        x1, x2, x3 = x
        ui, vi, wi = self.V[i]
        denom = vi*x2 + wi*x3
        return x1, x2, x3, ui, vi, wi, denom

    def fi(self, x, i):
        x1, x2, x3, ui, vi, wi, denom = self._vars(x, i)
        return self.y[i] - (x1 + ui/denom)

    def gradi(self, x, i):
        x1, x2, x3, ui, vi, wi, denom = self._vars(x, i)
        grd = np.zeros(self.n)
        grd[0] = -1.0
        grd[1] = ui * vi / denom**2
        grd[2] = ui * wi / denom**2
        return grd

    def hessi(self, x, i):
        x1, x2, x3, ui, vi, wi, denom = self._vars(x, i)
        hss = np.zeros((self.n, self.n))
        hss[1,1] = -2 * ui * vi**2 / denom**3
        hss[2,2] = -2 * ui * wi**2 / denom**3
        hss[2,1] = -2 * ui * vi * wi / denom**3
        return hss

    def third_order_tensori(self, x, i):
        x1, x2, x3, ui, vi, wi, denom = self._vars(x, i)
        trd = np.zeros((self.n, self.n, self.n))
        c = 6 * ui / denom**4
        trd[1,1,1] = c * vi**3
        trd[2,1,1] = c * wi * vi**2
        trd[2,2,1] = c * wi**2 * vi
        trd[2,2,2] = c * wi**3
        return trd

class Gaussian(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([0.4, 1.0, 0.0])
        m = 15
        y_base = [0.0, 0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989]
        self.y = [y_base[i] if i <= 8 else y_base[16 - i] for i in range(m+1)]
        self.t = [(8-i)/2 for i in range(m+1)]
        super().__init__(x0,m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        diff = self.t[i]-x3
        return x1,x2,x3,diff,np.exp(-x[1]*diff**2/2)

    def fi(self, x, i):
        val = x[0]
        x1,x2,x3,diff,exp = self.spevals(x,i)
        val = x1*exp - self.y[i]
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])
        x1,x2,x3,diff,exp = self.spevals(x,i)
        grd[0] = exp
        grd[1] = -x1*diff**2/2*exp
        grd[2] = x1*x2*diff*exp
        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])
        x1,x2,x3,diff,exp = self.spevals(x,i)

        hss[1,1] = x1*(diff**2/2)**2*exp
        hss[2,2] = (-x1*x2 + x1*(x2*diff)**2)*exp
        hss[1,0] = -diff**2/2*exp
        hss[2,0] = x2*diff*exp
        hss[2,1] = (x1*diff-x1*x2*diff**3/2)*exp
        
        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])
        x1,x2,x3,diff,exp = self.spevals(x,i)

        trd[1,1,1] = -x1*(diff**2/2)**3*exp
        trd[2,2,2] = ((-x1*x2 + x1*(x2*diff)**2)*(x2*diff) - 2*x1*x2**2*diff)*exp

        trd[1,1,0] = (diff**2/2)**2*exp
        trd[2,1,1] = ((x1*diff - x1*x2*diff**3/2)*(-diff**2/2) - x1*diff**3/2)*exp
        trd[2,2,0] = (-x2 + (x2*diff)**2)*exp
        trd[2,2,1] = ((-x1*x2 + x1*(x2*diff)**2)*(-diff**2/2) - x1 + 2*x1*x2*diff**2)*exp

        trd[2,1,0] = (diff-x2*diff**3/2)*exp

        return trd

class Meyer(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([0.02, 4000.0, 250.0])
        m = 16
        self.y = [0, 34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872]
        self.t = [45+5*i for i in range(m+1)]
        super().__init__(x0,m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        denom = 1 / (self.t[i] + x3)
        arg = x2 * denom
        return x1,x2,x3,denom,np.exp(x2*denom)

    def fi(self, x, i):
        val = x[0]
        x1,x2,x3,denom,exp = self.spevals(x,i)
        val = x1*exp - self.y[i]
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])
        x1,x2,x3,denom,exp = self.spevals(x,i)
        grd[0] = exp
        grd[1] = x1*denom*exp
        grd[2] = -x1*x2*denom**2*exp
        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])
        x1,x2,x3,denom,exp = self.spevals(x,i)

        hss[1,1] = x1*denom**2*exp
        hss[2,2] = (2*x1*x2*denom**3 + x1*x2**2*denom**4)*exp
        hss[1,0] = denom*exp
        hss[2,0] = -x2*denom**2*exp
        hss[2,1] = -x1*(denom**2 + x2*denom**3)*exp
        
        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])
        x1,x2,x3,denom,exp = self.spevals(x,i)

        trd[1,1,1] = x1*denom**3*exp
        trd[2,2,2] = ((2*x1*x2*denom**3 + x1*x2**2*denom**4)*(-x2*denom**2) - (6*x1*x2*denom**4 + 4*x1*x2**2*denom**5))*exp

        trd[1,1,0] = denom**2*exp
        trd[2,2,0] = (2*x2*denom**3 + x2**2*denom**4)*exp
        trd[2,2,1] = ((2*x1*x2*denom**4 + 2*x1*denom**3) + (x1*x2**2*denom**5 + 2*x1*x2*denom**4))*exp
        trd[2,1,0] = -(denom**2 + x2*denom**3)*exp
        trd[2,1,1] = (-2*x1*denom**3 - x1*x2*denom**4)*exp
       

        return trd
#11--
class GulfResearchAndDevelopement(ChainRule):
    def __init__(self, m=10, x0=None):
        if x0 is None:
            x0 = np.array([5.0, 2.5, 0.15])
        self.t = [i/100 for i in range(m+1)]
        self.y = [0] + [25 + (-50*np.log(self.t[i]))**(2/3) for i in range(1, m+1)]
        super().__init__(x0, m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        yi = self.y[i]
        diff = (yi-x2)
        frac = np.abs(diff)**x3/x1
        expo = np.exp(-frac)
        return x1,x2,x3,yi,diff,frac,expo

    def fi(self, x, i):
        val = x[0]
        x1,x2,x3,yi,diff,frac,expo = self.spevals(x,i)
        ti = self.t[i]
        yi = self.y[i]
        val = np.exp(-np.abs(yi-x2)**x3/x1) - ti
        return val        

    def gradi(self, x, i):
        n = np.shape(self.x0)[0]
        grd = np.zeros([n])
        x1,x2,x3,yi,diff,frac,expo = self.spevals(x,i)
        
        grd[0] = expo*frac/x1
        grd[1] = -x3*np.abs(diff)**(x3-1)*(-1)*np.sign(diff)*expo/x1
        grd[2] = -expo*np.log(np.abs(diff))*frac

        return grd
    
    def hessi(self, x, i):
        n = np.shape(self.x0)[0]
        hss = np.zeros([n,n])
        x1,x2,x3,yi,diff,frac,expo = self.spevals(x,i)
        sign = np.sign(diff)
        absv = np.abs(diff)
        hss[0,0] = expo*frac*(frac/x1-2/x1)/x1
        hss[1,0] = (x3*np.abs(diff)**(x3-1)*(-1)*sign/x1)*(1/x1 - frac/x1)*expo
        hss[1,1] = ((x3*absv**(x3-1)*(-1)*sign/x1)**2 - x3*(x3-1)*absv**(x3-2)*((-1))**2/x1)*expo     
        hss[2,0] = (np.log(np.abs(diff))*frac)*(1/x1 - frac/x1)*expo
        hss[2,1] = (frac*np.log(np.abs(diff))*x3*np.abs(diff)**(x3-1)/x1*sign*(-1) + (-x3*np.abs(diff)**(x3-1)/x1*sign*np.log(np.abs(diff))*(-1) - frac*(-1)/(diff)))*expo
        hss[2,2] = ((np.log(np.abs(diff))*frac)**2 - np.log(np.abs(diff))**2*frac)*expo
        
        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(self.x0)[0]
        trd = np.zeros([n,n,n])
        x1,x2,x3,yi,diff,frac,expo = self.spevals(x,i)
        absv = np.abs(diff)
        sign = np.sign(diff)

        trd[0,0,0] = (0
            + 2*(absv**x3/x1**2)*absv**x3/x1**3*(-2) 
            + 2*absv**x3*3/x1**4
            + ((absv**x3/x1**2)**2 - 2*absv**x3/x1**3)*(absv**x3/x1**2)
        )*expo

        trd[1,0,0] = x3 * absv**(x3-1) * sign * (-1) * expo * (4*frac - 2 - frac**2) / x1**3

        trd[2,0,0] = np.log(absv) * expo * (4*frac**2 - 2*frac - frac**3) / x1**2

        trd[1,1,0] = expo * (
            (x3*absv**(x3-1)*sign*(-1))**2 * (frac - 2)/x1**3 
            + x3*(x3-1)*absv**(x3-2)*((-1))**2 * (1 - frac)/x1**2
        )

        trd[2,1,0] = expo * (
            x3 * absv**(x3-1) * sign * (-1) * np.log(absv) * (frac**2 - 3*frac + 1) / x1**2
            + absv**(x3-1) * sign * (frac - 1) / x1**2
        )

        trd[2,2,0] = expo * (np.log(absv))**2 * frac * (1 - 3*frac + frac**2) / x1

        trd[1,1,1] = expo * (
            3 * x3 * absv**(x3-1) * sign * (-1) * 
            x3*(x3-1) * absv**(x3-2) * ((-1))**2 / x1**2
            - x3*(x3-1)*(x3-2) * absv**(x3-3) * sign * ((-1))**3 / x1
            - (x3 * absv**(x3-1) * sign * (-1))**3 / x1**3
        )

        trd[2,1,1] = expo * (
            (2 * (x3 * absv**(x3-1) * sign * (-1))**2 / x1**2) * (1/x3 + np.log(absv))
            - (x3 * (x3-1) * absv**(x3-2) * ((-1))**2 / x1) * ((2*x3 - 1)/(x3*(x3 - 1)) + np.log(absv))
            - ((x3 * absv**(x3-1) * sign * (-1))**2 / x1**2 - x3 * (x3-1) * absv**(x3-2) * ((-1))**2 / x1) * (absv**x3 / x1) * np.log(absv)
        )

        trd[2,2,1] = expo * (
            2*np.log(absv)*(sign*(-1)/absv)*(frac**2 - frac) 
            + (np.log(absv)**2)*(x3/absv)*frac*sign*(-1)*(3*frac - 1 - frac**2)
        )

        trd[2,2,2] = expo * (np.log(absv))**3 * frac * (3*frac - 1 - frac**2)

        return trd

class Box3D(ChainRule):
    def __init__(self, m=10, x0=None):
        if x0 is None:
            x0 = np.array([0.0, 10.0, 20.0])
        self.t = [i/10 for i in range(m+1)]
        super().__init__(x0, m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x1,x2,x3

    def fi(self, x, i):
        val = x[0]
        x1,x2,x3 = self.spevals(x,i)
        ti = self.t[i]
        val = np.exp(-ti*x1) - np.exp(-ti*x2) - x3*(np.exp(-ti)-np.exp(-ti*10))
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])
        x1,x2,x3 = self.spevals(x,i)
        ti = self.t[i]
        grd[0] = -ti*np.exp(-ti*x1)
        grd[1] = ti*np.exp(-ti*x2)
        grd[2] = -np.exp(-ti)+np.exp(-ti*10)
        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])
        x1,x2,x3 = self.spevals(x,i)
        ti = self.t[i]
        hss[0,0] = ti**2*np.exp(-ti*x1)
        hss[1,1] = -ti**2*np.exp(-ti*x2)
        
        
        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])
        x1,x2,x3 = self.spevals(x,i)
        ti = self.t[i]
        trd[0,0,0] = -ti**3*np.exp(-ti*x1)
        trd[1,1,1] = ti**3*np.exp(-ti*x2)

        return trd

class PowellSingular(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([3.0, -1.0, 0.0, 1.0])
        super().__init__(x0, 4)

    def spevals(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        return x1,x2,x3,x4
    
    def f(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        val = 0.0
        val += (x1 + 10*x2)**2
        val += 5*(x3 - x4)**2
        val += (x2 - 2*x3)**4
        val += 10*(x1 - x4)**4
        return val
    
    def grad(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        grd = np.zeros([self.n])

        grd[0] += 2*x1 + 20*x2
        grd[1] += 200*x2 + 20*x1

        grd[2] += 10*x3 - 10*x4
        grd[3] += 10*x4 - 10*x3

        grd[1] += 4*(x2 - 2*x3)**3
        grd[2] += -8*(x2 - 2*x3)**3

        grd[0] += 40*(x1 - x4)**3
        grd[3] += -40*(x1 - x4)**3
        return grd
    def hess(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        hss = np.zeros([self.n,self.n])

        hss[0,0] += 2
        hss[1,1] += 200
        hss[1,0] += 20
        hss[0,1] += 20

        hss[2,2] += 10
        hss[3,3] += 10
        hss[3,2] += -10
        hss[2,3] += -10

        hss[1,1] += 12*(x2 - 2*x3)**2
        hss[2,2] += 48*(x2 - 2*x3)**2
        hss[2,1] += -24*(x2 - 2*x3)**2
        hss[1,2] += -24*(x2 - 2*x3)**2

        hss[0,0] += 120*(x1 - x4)**2
        hss[3,3] += 120*(x1 - x4)**2
        hss[3,0] += -120*(x1 - x4)**2
        hss[0,3] += -120*(x1 - x4)**2
        
        return hss
    def third_order_tensor(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        trd = np.zeros([self.n,self.n,self.n])

        trd[1,1,1] += 24*(x2 - 2*x3)
        trd[2,2,1] += 96*(x2 - 2*x3)
        trd[2,1,1] += -48*(x2 - 2*x3)
        trd[2,2,2] += -192*(x2 - 2*x3)
        trd[2,1,2] = trd[2,2,1]
        trd[1,2,2] = trd[2,2,1]
        trd[1,2,1] = trd[2,1,1]
        trd[1,1,2] = trd[2,1,1]


        trd[0,0,0] += 240*(x1 - x4)
        trd[3,3,0] += 240*(x1 - x4)
        trd[3,0,0] += -240*(x1 - x4)
        trd[3,3,3] += -240*(x1 - x4)
        trd[3,0,3] = trd[3,3,0]
        trd[0,3,3] = trd[3,3,0]
        trd[0,3,0] = trd[3,0,0]
        trd[0,0,3] = trd[3,0,0]

        
        return trd

class Wood(TestFunction):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([-3.0, -1.0, -3.0, -1.0])
        super().__init__(x0, 6)

    def spevals(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        return x1,x2,x3,x4
    
    def f(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        val = 0.0
        val += 100*(x2 - x1**2)**2
        val += (1 - x1)**2
        val += 90*(x4 - x3**2)**2
        val += (1 - x3)**2
        val += 10*(x2 + x4 - 2)**2
        val += .1*(x2 - x4)**2
        return val
    
    def grad(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        grd = np.zeros([self.n])

        grd[0] += 400*x1**3 - 400*x1*x2
        grd[1] += 200*x2 - 200*x1**2

        grd[0] += 2*x1 - 2

        grd[2] += 90*(4*x3**3 - 4*x3*x4)
        grd[3] += 90*(2*x4 - 2*x3**2)

        grd[2] += 2*x3 - 2

        grd[1] += 20*(x2 + x4 - 2)
        grd[3] += 20*(x2 + x4 - 2)

        grd[1] += .2*x2 - .2*x4
        grd[3] += .2*x4 - .2*x2

        return grd
    
    def hess(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        hss = np.zeros([self.n,self.n])

        hss[0,0] += 1200*x1**2 - 400*x2
        hss[1,1] += 200
        hss[1,0] += -400*x1
        hss[0,1] += -400*x1

        hss[0,0] += 2

        hss[2,2] += 90*(12*x3**2 - 4*x4)
        hss[3,2] += -360*x3
        hss[2,3] += -360*x3
        hss[3,3] += 180

        hss[2,2] += 2

        hss[1,3] += 20
        hss[3,1] += 20
        hss[1,1] += 20
        hss[3,3] += 20

        hss[1,1] += .2
        hss[3,3] += .2
        hss[1,3] += -.2
        hss[3,1] += -.2
        
        return hss
    def third_order_tensor(self, x):
        x1,x2,x3,x4 = self.spevals(x)
        trd = np.zeros([self.n,self.n,self.n])

        trd[0,0,0] += 2400*x1
        trd[1,0,0] += -400
        trd[0,1,0] = trd[1,0,0]
        trd[0,0,1] = trd[1,0,0]

        trd[2,2,2] += 90*24*x3
        trd[3,2,2] += -360
        trd[2,3,2] = trd[3,2,2]
        trd[2,2,3] =trd[3,2,2]
     
        return trd

class KowalikAndOsborne(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([0.25, 0.39, 0.415, 0.39])
        m = 11
        self.y = [0.0, 0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
        self.u = [0.0, 4.0000, 2.0000, 1.0000, 0.5000, 0.2500, 0.1670, 0.1250, 0.1000, 0.0833, 0.0714, 0.0625]
        super().__init__(x0,m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        yi = self.y[i]
        ui = self.u[i]
        paren = (ui**2 + ui*x2)
        denom = 1/(ui**2 + ui*x3 + x4)
        return x1,x2,x3,x4,yi,ui,paren,denom

    def fi(self, x, i):
        val = x[0]
        x1,x2,x3,x4,yi,ui,paren,denom = self.spevals(x,i)
        val = yi - x1*paren*denom
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        x1,x2,x3,x4,yi,ui,paren,denom = self.spevals(x,i)

        grd[0] = -paren*denom
        grd[1] = -x1*ui*denom
        grd[2] = x1*ui*paren*denom**2
        grd[3] = x1*paren*denom**2

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        x1,x2,x3,x4,yi,ui,paren,denom = self.spevals(x,i)

        for j in range(4) :
            for k in range(j+1) :
                numbers = np.array([0,0,0,0],dtype=int)
                numbers[j] += 1
                numbers[k] += 1
                numer = 1.0
                if numbers[0] == 2 or numbers[1] == 2 :
                    numer *= 0.0
                if numbers[0] == 0 :
                    numer *= x1
                if numbers[1] == 0:
                    numer *= (ui**2 + ui*x2)
                else:
                    numer *= ui
                numer *= ui**numbers[2]
                numer *=(-denom)**(numbers[2] + numbers[3] + 1)
                if numbers[2] + numbers[3] == 2 :
                    numer *= 2
                
                hss[j,k] += numer
        
        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        x1,x2,x3,x4,yi,ui,paren,denom = self.spevals(x,i)

        for j in range(4) :
            for k in range(j+1) :
                for l in range(k+1):
                    numbers = np.array([0,0,0,0],dtype=int)
                    numbers[j] += 1
                    numbers[k] += 1
                    numbers[l] += 1
                    numer = 1.0
                    if numbers[0] >= 2 or numbers[1] >= 2 :
                        numer *= 0.0
                    if numbers[0] == 0 :
                        numer *= x1
                    if numbers[1] == 0:
                        numer *= (ui**2 + ui*x2)
                    else:
                        numer *= ui
                    numer *= ui**numbers[2]
                    numer *=(-denom)**(numbers[2] + numbers[3] + 1)
                    if numbers[2] + numbers[3] == 2 :
                        numer *= 2
                    if numbers[2] + numbers[3] == 3 :
                        numer *= 6
                    
                    trd[j,k,l] += numer

        return trd

class BrownAndDennis(ChainRule):
    def __init__(self, m=20, x0=None):
        if x0 is None:
            x0 = np.array([25., 5., -5., -1.])
        self.t = [i/5 for i in range(m+1)]
        super().__init__(x0, m)

    def spevals(self, x, i):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        ti = self.t[i]
        return x1,x2,x3,x4,ti

    def fi(self, x, i):
        x1,x2,x3,x4,ti = self.spevals(x,i)
        val = (x1 + ti*x2 - np.exp(ti))**2 + (x3 + x4*np.sin(ti) - np.cos(ti))**2
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])
        x1,x2,x3,x4,ti = self.spevals(x,i)

        grd[0] = 2*(x1 + ti*x2 - np.exp(ti))
        grd[1] = grd[0]*ti
        grd[2] = 2*(x3 + x4*np.sin(ti) - np.cos(ti))
        grd[3] = grd[2]*np.sin(ti)

        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])
        x1,x2,x3,x4,ti = self.spevals(x,i)

        hss[0,0] = 2
        hss[1,0] = 2*ti
        hss[1,1] = 2*ti**2
        hss[2,2] = 2
        hss[3,2] = 2*np.sin(ti)
        hss[3,3] = 2*np.sin(ti)**2
        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])
        return trd

class Osborne1(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([0.5, 1.5, -1.0, 0.01, 0.02])
        m = 33
        self.t = [10*(i-1) for i in range(m+1)]
        self.y = [0.0, 0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406]
        super().__init__(x0,m)

    def fi(self, x, i):
        val = self.y[i] - (x[0] + x[1]*np.exp(-self.t[i]*x[3]) + x[2]*np.exp(-self.t[i]*x[4]))
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])

        grd[0] = -1
        pairs = ((2,4),(3,5))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            grd[a] = -np.exp(-self.t[i]*x[b])
            grd[b] = -grd[a]*self.t[i]*x[a]

        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])

        pairs = ((2,4),(3,5))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            hss[b,a] = self.t[i]*np.exp(-self.t[i]*x[b])
            hss[b,b] = -hss[b,a]*self.t[i]*x[a]

        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])

        pairs = ((2,4),(3,5))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            trd[b,b,a] = -self.t[i]**2*np.exp(-self.t[i]*x[b])
            trd[b,b,b] = -trd[b,b,a]*self.t[i]*x[a]

        return trd

class BiggsEXP6(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([1, 2, 1, 1, 1, 1], dtype=float)
        m = 13
        self.t = [.1*i for i in range(m+1)]
        self.y = [np.exp(-self.t[i]) - 5*np.exp(-10*self.t[i]) + 3*np.exp(-4*self.t[i]) for i in range(m+1)]        
        super().__init__(x0,m)

    def fi(self, x, i):
        val = -self.y[i]
        sign = 1.0
        pairs = ((1,3),(2,4),(5,6))
        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            val += sign*x[b]*np.exp(-self.t[i]*x[a])
            sign *= -1.0

        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])

        sign = 1.0
        pairs = ((1,3),(2,4),(5,6))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            grd[b] = sign*np.exp(-self.t[i]*x[a])
            grd[a] = -grd[b]*self.t[i]*x[b]
            sign *= -1

        return grd
    
    def hessi(self, x, i):
        hss = np.zeros([self.n,self.n])

        sign = 1.0
        pairs = ((1,3),(2,4),(5,6))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            hss[b,a] = -sign*np.exp(-self.t[i]*x[a])*self.t[i]
            hss[a,a] = -hss[b,a]*self.t[i]*x[b]
            sign *= -1

        return hss
    
    def third_order_tensori(self, x, i):
        trd = np.zeros([self.n,self.n,self.n])

        sign = 1.0
        pairs = ((1,3),(2,4),(5,6))

        for pair in pairs :
            a = pair[0] - 1
            b = pair[1] - 1
            trd[b,a,a] = sign*np.exp(-self.t[i]*x[a])*self.t[i]**2
            trd[a,a,a] = -trd[b,a,a]*self.t[i]*x[b]
            sign *= -1

        return trd

class Osborne2(ChainRule):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])
        m = 65
        self.t = [(i-1)/10 for i in range(m+1)]
        self.y = [0.0, 1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]
        super().__init__(x0,m)

    def fi(self, x, i):
        val = self.y[i]
        triplets = ((2,6,9),(3,7,10),(4,8,11))

        a = 1 - 1
        b = 5 - 1

        val += -x[a]*np.exp(-self.t[i]*x[b])

        for triplet in triplets :
            a = triplet[0] - 1
            b = triplet[1] - 1
            c = triplet[2] - 1
            val += - x[a]*np.exp(-(self.t[i]-x[c])**2*x[b])
        
        return val        

    def gradi(self, x, i):
        grd = np.zeros([self.n])

        triplets = ((2,6,9),(3,7,10),(4,8,11))

        a = 1 - 1
        b = 5 - 1

        grd[a] = -np.exp(-self.t[i]*x[b])
        grd[b] = -x[a]*grd[a]*self.t[i]

        for triplet in triplets :
            a = triplet[0] - 1
            b = triplet[1] - 1
            c = triplet[2] - 1
            grd[a]= - np.exp(-(self.t[i]-x[c])**2*x[b])
            grd[b]= - grd[a]*x[a]*(self.t[i]-x[c])**2
            grd[c]= 2*grd[a]*x[a]*x[b]*(self.t[i]-x[c])

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])

        triplets = ((2,6,9),(3,7,10),(4,8,11))

        a = 1 - 1
        b = 5 - 1

        hss[b,a] = np.exp(-self.t[i]*x[b])*self.t[i]
        hss[b,b] = -hss[b,a]*self.t[i]*x[a]

        for triplet in triplets :
            a = triplet[0] - 1
            b = triplet[1] - 1
            c = triplet[2] - 1
            delta = (self.t[i]-x[c])
            expbc = np.exp(-delta**2*x[b])
            hss[b,a] = expbc*delta**2
            hss[b,b] = -hss[b,a]*delta**2*x[a]
            hss[c,a] = -2*expbc*x[b]*delta
            hss[c,b] = -(2*delta*x[a] - 2*x[a]*x[b]*delta**3)*expbc
            hss[c,c] = -(-2*x[a]*x[b] + 4*x[a]*x[b]**2*delta**2)*expbc
            

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        triplets = ((2,6,9),(3,7,10),(4,8,11)) #1,5,8 2,6,9 3,7,10

        a = 1 - 1 #0,4
        b = 5 - 1

        trd[b,b,a] = -self.t[i]**2*np.exp(-self.t[i]*x[b])
        trd[b,b,b] = -trd[b,b,a]*self.t[i]*x[a]

        for triplet in triplets :
            a = triplet[0] - 1
            b = triplet[1] - 1
            c = triplet[2] - 1
            delta = (self.t[i]-x[c])
            expbc = np.exp(-delta**2*x[b])

            trd[b,b,a] = -delta**4*expbc
            trd[b,b,b] = delta**6*expbc*x[a]
            trd[c,b,a] = -(2*delta - 2*x[b]*delta**3)*expbc
            trd[c,c,a] = -(-2*x[b] + 4*x[b]**2*delta**2)*expbc
            trd[c,b,b] = -(-4*delta**3*x[a] + 2*x[a]*x[b]*delta**5)*expbc
            trd[c,c,b] = -(-2*x[a] + 8*x[a]*x[b]*delta**2 - (4*x[a]*x[b]**2*delta**2 - 2*x[a]*x[b])*delta**2)*expbc
            trd[c,c,c] = -(-8*x[a]*x[b]**2*delta + (-2*x[a]*x[b] + 4*x[a]*x[b]**2*delta**2)*2*x[b]*delta)*expbc

        return trd

class Watson(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 6
            x0 = np.zeros(n)
        m = 31
        self.t = [i/29 for i in range(m+1)]
        super().__init__(x0, m)

    def fi(self, x, i):
        n = np.shape(x)[0]

        if i < 30 :
            val = -1.0

            for j in range(1,n) :
                val += j*x[j]*self.t[i]**(j-1)

            sumsq = 0.0

            for j in range(0,n) :
                sumsq += x[j]*self.t[i]**j

            val += -sumsq**2

        elif i == 30 :
            val = x[0]
        
        elif i == 31 :
            val = x[1] - x[0]**2 - 1.0
        
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        if i < 30 :
            for j in range(1,n) :
                grd[j] += j*self.t[i]**(j-1)

            for k in range(0,n) :
                sum = 0.0
                for j in range(0,n) :
                    sum += x[j]*self.t[i]**j
                grd[k] += -2*self.t[i]**k*sum
            
        elif i == 30 :
            grd[0] += 1

        elif i == 31 :
            grd[0] += -2*x[0]
            grd[1] += 1

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n]) 

        if i < 30 :
            for k in range(0,n) :
                for l in range(0,k+1) :
                    hss[k,l] += -2*self.t[i]**k*self.t[i]**l

        elif i == 31 :
            hss[0,0] = -2

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        return trd
#21--
class ExtendedRosenbrock(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.array([-1.2 if (i % 2 == 0) else 1.0 for i in range(n)])
        super().__init__(x0, n)

    def f(self, x):
        n = np.shape(x)[0]
        val = 0.0

        for i in range(n // 2): 
            val += (10 * (x[2*i+1] - x[2*i]**2))**2 + (1 - x[2*i])**2
        
        return val         

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        for i in range(n // 2):
            grd[2*i] += 2*x[2*i] - 2
            grd[2*i] += 400*x[2*i]**3 - 400*x[2*i]*x[2*i+1]
            grd[2*i+1] += 200*x[2*i+1] - 200*x[2*i]**2

        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n, n])

        for i in range(n // 2): 
            hss[2*i][2*i] += 2
            hss[2*i][2*i] += 1200*x[2*i]**2 - 400*x[2*i+1]
            hss[2*i+1][2*i] += -400*x[2*i]
            hss[2*i][2*i+1] += -400*x[2*i]
            hss[2*i+1, 2*i+1] += 200

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n, n, n])

        for i in range(n // 2): 
            trd[2*i][2*i][2*i] += 2400*x[2*i]
            trd[2*i+1][2*i][2*i] += -400
            trd[2*i][2*i+1][2*i] += -400
            trd[2*i][2*i][2*i+1] += -400

        return trd
    
class ExtendedPowellSingular(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 12
            pattern = [3.0, -1.0, 0.0, 1.0]
            x0 = np.array([pattern[i % 4] for i in range(n)])
        super().__init__(x0, n)

    def f(self, x):
        n = len(x)
        val = 0.0
        # Loop through blocks of 4
        for i in range(n // 4): 
            x1, x2, x3, x4 = x[4*i], x[4*i+1], x[4*i+2], x[4*i+3]
            
            f1 = x1 + 10*x2
            f2 = np.sqrt(5) * (x3 - x4)
            f3 = (x2 - 2*x3)**2
            f4 = np.sqrt(10) * (x1 - x4)**2
            
            val += f1**2 + f2**2 + f3**2 + f4**2
        return val

    def grad(self, x):
        n = len(x)
        grd = np.zeros(n)
        for i in range(n // 4):
            x1, x2, x3, x4 = x[4*i], x[4*i+1], x[4*i+2], x[4*i+3]
            
            # d/dx1
            grd[4*i]   = 2*(x1 + 10*x2) + 40*(x1 - x4)**3
            # d/dx2
            grd[4*i+1] = 20*(x1 + 10*x2) + 4*(x2 - 2*x3)**3
            # d/dx3
            grd[4*i+2] = 10*(x3 - x4) - 8*(x2 - 2*x3)**3
            # d/dx4
            grd[4*i+3] = -10*(x3 - x4) - 40*(x1 - x4)**3
        return grd

    def hess(self, x):
        n = len(x)
        hss = np.zeros([n, n])
        for i in range(n // 4):
            x1, x2, x3, x4 = x[4*i], x[4*i+1], x[4*i+2], x[4*i+3]
            
            # Diagonal elements
            hss[4*i,   4*i]   = 2 + 120*(x1 - x4)**2
            hss[4*i+1, 4*i+1] = 200 + 12*(x2 - 2*x3)**2
            hss[4*i+2, 4*i+2] = 10 + 48*(x2 - 2*x3)**2
            hss[4*i+3, 4*i+3] = 10 + 120*(x1 - x4)**2
            
            # Off-diagonal elements (Symmetric)
            hss[4*i,   4*i+1] = hss[4*i+1, 4*i]   = 20
            hss[4*i,   4*i+3] = hss[4*i+3, 4*i]   = -120*(x1 - x4)**2
            hss[4*i+1, 4*i+2] = hss[4*i+2, 4*i+1] = -24*(x2 - 2*x3)**2
            hss[4*i+2, 4*i+3] = hss[4*i+3, 4*i+2] = -10
        return hss
    def third_order_tensor(self, x): 
        n = np.shape(x)[0] 
        trd = np.zeros([n,n,n]) 

        for i in range(n//4):  
            trd[4*i][4*i][4*i] += 240*(x[4*i] - x[4*i+3]) 
            trd[4*i+1][4*i+1][4*i+1] += 24*(x[4*i+1] - 2*x[4*i+2]) 
            trd[4*i+3][4*i+3][4*i+3] += -240*(x[4*i] - x[4*i+3]) 
            trd[4*i+2][4*i+2][4*i+2] += -192*(x[4*i+1] - 2*x[4*i+2]) 

            trd[4*i+1][4*i+2][4*i+2] += 96*(x[4*i+1] - 2*x[4*i+2]) 
            trd[4*i+2][4*i+1][4*i+2] += 96*(x[4*i+1] - 2*x[4*i+2]) 
            trd[4*i+2][4*i+2][4*i+1] += 96*(x[4*i+1] - 2*x[4*i+2]) 

            trd[4*i+2][4*i+1][4*i+1] += -48*(x[4*i+1] - 2*x[4*i+2]) 
            trd[4*i+1][4*i+2][4*i+1] += -48*(x[4*i+1] - 2*x[4*i+2]) 
            trd[4*i+1][4*i+1][4*i+2] += -48*(x[4*i+1] - 2*x[4*i+2]) 

            trd[4*i+3][4*i][4*i] += -240*(x[4*i] - x[4*i+3]) 
            trd[4*i][4*i+3][4*i] += -240*(x[4*i] - x[4*i+3]) 
            trd[4*i][4*i][4*i+3] += -240*(x[4*i] - x[4*i+3]) 

            trd[4*i][4*i+3][4*i+3] += 240*(x[4*i] - x[4*i+3]) 
            trd[4*i+3][4*i][4*i+3] += 240*(x[4*i] - x[4*i+3]) 
            trd[4*i+3][4*i+3][4*i] += 240*(x[4*i] - x[4*i+3]) 

        return trd

class PenaltyI(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 4
            x0 = np.array([float(j) for j in range(n)])
        super().__init__(x0, n + 1)

    def f(self, x):
        n = np.shape(x)[0]
        a = 1e-5
        val = 0.0
        sum = 0.0

        for i in range(self.m-1) : 
            val += a*(x[i] - 1)**2
            sum += x[i]**2

        val += (sum - 1/4)**2
        
        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        sum = 0.0
        a = 1e-5
        for i in range(self.m-1) : 
            sum += x[i]**2
        
        for k in range(self.m-1) :
            grd[k] += 4*x[k]*(sum - 1/4)
            grd[k] += 2*a*x[k] - 2*a
        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        sum = 0.0
        a = 1e-5

        for i in range(self.m - 1) : 
            sum += x[i]**2

        for k in range(self.m - 1) :
            hss[k,k] += 4*sum - 1 + 2*a
            for l in range(self.m - 1) :
                hss[k][l] += 8*x[k]*x[l]

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        a = 1e-5

        for k in range(self.m - 1) :
            for l in range(k,self.m - 1) :
                trd[k,k,l] += 8*x[l]
                trd[k,l,k] += 8*x[l]
                trd[l,k,k] += 8*x[l]
                trd[l,l,k] += 8*x[k]
                trd[l,k,l] += 8*x[k]
                trd[k,l,l] += 8*x[k]
            trd[k,k,k] = 24*x[k]


        return trd

class PenaltyII(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 4
            x0 = np.ones(n) * 0.5
        m = 2 * n
        self.y = np.array([np.exp(i/10) + np.exp((i-1)/10) for i in range(1, m+1)])
        super().__init__(x0, m)

    def f(self, x):
        n = np.shape(x)[0]
        val = (x[0] - .2)**2
        sum = 0.0

        a = 1e-5

        for i in range(1,n) : 
            val += a*(np.exp(x[i]/10) + np.exp(x[i-1]/10 )- self.y[i])**2
            val += a*(np.exp(x[i]/10) - np.exp(-1/10))**2
        for j in range(n) :
            sum += (n-j)*x[j]**2
        val += (sum-1)**2
        
        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        sum = 0.0
        a = 1e-5
        grd[0] += 2*(x[0] - .2)
        for j in range(n) :
            sum += (n-j)*x[j]**2
        for i in range(1,n) : 
            grd[i] += .2*a*(np.exp(x[i]/10) + np.exp(x[i-1]/10) - self.y[i])*np.exp(x[i]/10)
            grd[i-1] += .2*a*(np.exp(x[i]/10) + np.exp(x[i-1]/10) - self.y[i])*np.exp(x[i-1]/10)
            grd[i] += .2*a*(np.exp(x[i]/10) - np.exp(-1/10))*np.exp(x[i]/10)
        for i in range(n) : 
            grd[i] += 4*(n-i)*x[i]*(sum-1)

        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        sum = 0.0
        a = 1e-5
        hss[0,0] += 2.0
        for j in range(n) :
            sum += (n-j)*x[j]**2
        for i in range(1,n) : 
            hss[i,i] += .2*a*(.2*np.exp(x[i]/10)**2 + .1*np.exp(x[i-1]/10)*np.exp(x[i]/10) - .1*self.y[i]*np.exp(x[i]/10))
            hss[i,i-1] += .2*a*(.1*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            hss[i-1,i] += .2*a*(.1*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            hss[i-1,i-1] += .2*a*(.1*np.exp(x[i-1]/10)*np.exp(x[i]/10) + .2*np.exp(x[i-1]/10)**2 - .1*self.y[i]*np.exp(x[i-1]/10))
            hss[i,i] += .2*a*(.2*np.exp(x[i]/10)**2 - .1*np.exp(-1/10)*np.exp(x[i]/10))
        for i in range(n) : 
            hss[i,i] += 4*(n-i)*(sum-1) + 8*(n-i)**2*x[i]**2
            for j in range(i+1,n) :
                hss[i,j] += 8*(n-i)*(n-j)*x[i]*x[j]
                hss[j,i] += 8*(n-i)*(n-j)*x[i]*x[j]

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        sum = 0.0
        a = 1e-5
        for j in range(n) :
            sum += (n-j)*x[j]**2
        for i in range(1,n) : 
            trd[i,i,i] += .2*a*(.04*np.exp(x[i]/10)**2 + .01*np.exp(x[i-1]/10)*np.exp(x[i]/10) - .01*self.y[i]*np.exp(x[i]/10))
            trd[i,i-1,i-1] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i-1,i,i-1] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i-1,i-1,i] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i,i,i-1] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i,i-1,i] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i-1,i,i] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10))
            trd[i-1,i-1,i-1] += .2*a*(.01*np.exp(x[i-1]/10)*np.exp(x[i]/10) + .04*np.exp(x[i-1]/10)**2 - .01*self.y[i]*np.exp(x[i-1]/10))
            trd[i,i,i] += .2*a*(.04*np.exp(x[i]/10)**2 - .01*np.exp(-1/10)*np.exp(x[i]/10))
        for i in range(n) : 
            trd[i,i,i] += 8*(n-i)**2*x[i] + 16*(n-i)**2*x[i]
            for j in range(i+1,n) :
                trd[i,j,j] += 8*(n-i)*(n-j)*x[i]
                trd[j,i,j] += 8*(n-i)*(n-j)*x[i]
                trd[j,j,i] += 8*(n-i)*(n-j)*x[i]
                trd[j,i,i] += 8*(n-i)*(n-j)*x[j]
                trd[i,j,i] += 8*(n-i)*(n-j)*x[j]
                trd[i,i,j] += 8*(n-i)*(n-j)*x[j]

    

        return trd

class VariablyDimensioned(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.array([-1.2 if (i % 2 == 0) else 1.0 for i in range(n)])
        super().__init__(x0, n + 2)
    def f(self, x):
        n = np.shape(x)[0]
        val = 0.0
        sum = 0.0

        for i in range(n):
            val += (x[i] - 1)**2
            sum += (i+1)*(x[i] - 1)

        val+= sum**2 + sum**4
        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        sum = 0.0

        for i in range(n):
            sum += (i+1)*(x[i] - 1)

        for i in range(n):
            grd[i] += 2*x[i] - 2
            grd[i] += 2*(i+1)*sum
            grd[i] += 4*(i+1)*sum**3

        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        sum = 0.0

        for i in range(n):
            sum += (i+1)*(x[i] - 1)

        hss = np.zeros([n,n])
        for i in range(n):
            hss[i,i] += 2
            for j in range(n):
                hss[i,j] += 2*(i+1)*(j+1)
                hss[i,j] += 12*(i+1)*(j+1)*sum**2

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        sum = 0.0

        for i in range(n):
            sum += (i+1)*(x[i] - 1)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    trd[i,j,k] += 24*(i+1)*(j+1)*(k+1)*sum

        return trd

class Trigonometric(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n) / n
        super().__init__(x0, n)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = n + i*(1-np.cos(x[i-1])) - np.sin(x[i-1])

        for j in range(n):
            val -= np.cos(x[j])
       
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        for j in range(n):
            grd[j] += np.sin(x[j])
        
        grd[i-1] += i*np.sin(x[i-1]) - np.cos(x[i-1])

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])

        for j in range(n):
            hss[j,j] += np.cos(x[j])
        
        hss[i-1,i-1] += i*np.cos(x[i-1]) + np.sin(x[i-1])

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        for j in range(n):
            trd[j,j,j] += -np.sin(x[j])
        
        trd[i-1,i-1,i-1] += -i*np.sin(x[i-1]) + np.cos(x[i-1])

        return trd

class BrownAlmostLinear(TestFunction):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 40
            x0 = np.ones(n) / 2
        super().__init__(x0, n)

    def f(self, x):
        n = np.shape(x)[0]
        val = 0.0
        sum = 0.0
        prod = 1.0

        for j in range(n):
            sum += x[j]
            prod *= x[j]

        for i in range(n-1) : 
            val += (x[i] + sum - n - 1)**2

        val += (prod - 1)**2

        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        sum = 0.0
        prod = 1.0

        for j in range(n):
            sum += x[j]
            prod *= x[j]

        for i in range(n-1):
            grd[i] += 2*(x[i] + sum - n - 1)
            grd += 2*(x[i] + sum - n - 1)

        prod = np.prod(x)
        factor = 2 * (prod - 1)
        
        grad = np.zeros_like(x)
        for i in range(n):
            prod_without_i = np.prod(np.delete(x, i))
            grd[i] += 2 * (prod - 1) * prod_without_i

        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        prod = 1.0

        for j in range(n):
            prod *= x[j]

        for i in range(n-1):
            hss[i,i] += 4
            for k in range(n):
                for j in range(n):
                    if k == i or j == i :
                        hss[j,k] += 2
                    hss[j,k] += 2
        
        prod_without = []
        for i in range(n):
            prod_without.append(np.prod(np.delete(x, i)))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    prod_without_ij = prod_without[i]
                    hss[i, i] += 2 * (prod_without[i])**2
                else :
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    prod_without_ij = np.prod(x[mask])
                    hss[i, j] += 2 * (prod_without[i] * prod_without[j] + (prod - 1) * prod_without_ij)

        return hss
    
    
    def third_order_tensor(self,x):
        n = len(x)
        prod = np.prod(x)
        trd = np.zeros((n, n, n))
        
        prod_without = []
        for i in range(n):
            prod_without.append(np.prod(np.delete(x, i)))
        
        prod_without_ij = {}
        prod_without_ijk = {}
        
        for i in range(n):
            for j in range(n):
                mask_ij = np.ones(n, dtype=bool)
                mask_ij[i] = False
                mask_ij[j] = False
                prod_without_ij[(i, j)] = np.prod(x[mask_ij])
                
                for k in range(n):
                    mask_ijk = mask_ij.copy()
                    mask_ijk[k] = False
                    prod_without_ijk[(i, j, k)] = np.prod(x[mask_ijk])
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i == j == k:
                        trd[i, i, i] = 0
                    elif i == j and i != k:
                        trd[i, i, k] = 4 * prod_without[i] * prod_without_ij[(i, k)]
                    elif i == k and i != j:
                        trd[i, j, i] = 4 * prod_without[i] * prod_without_ij[(i, j)]
                    elif j == k and i != j:
                        trd[i, j, j] = 4 * prod_without[j] * prod_without_ij[(i, j)]
                    else:
                        trd[i, j, k] = 8 * prod_without_ij[(j, k)] * prod_without[i] - 2 * prod_without_ijk[(i, j, k)]
        
        return trd

class DiscreteBoundaryValue(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n) / n
        h = 1 / (n + 1)
        self.h = h
        self.t = np.array([(i + 1) * h for i in range(n)])
        super().__init__(x0, n)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = 0.0
        i-=1
        if i != n-1:
            val += - x[i+1]
        if i != 0:
            val += - x[i-1]
        val += 2*x[i] + self.h**2*(x[i] + self.t[i] + 1)**3/2
       
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        i-=1
        if i != n-1:
            grd[i+1] += -1.0
        if i != 0:
            grd[i-1] += -1.0
        
        
        grd[i] += 2 + 3/2*self.h**2*(x[i] + self.t[i] + 1)**2
        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        i -= 1

        hss[i,i] += 3*self.h**2*(x[i] + self.t[i] + 1)

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        i -= 1

        trd[i,i,i] += 3*self.h**2

        return trd

class DiscreteIntegralEquation(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n) / n
        h = 1 / (n + 1)
        self.h = h
        self.t = np.array([(i + 1) * h for i in range(n)])
        super().__init__(x0, n)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = 0.0
        i-=1
        sum1 = 0.0
        sum2 = 0.0
        for j in range(i):
            sum1+=self.t[j]*(x[j]+self.t[j]+1)**3
        for j in range(i,n):
            sum2+=(1-self.t[j])*(x[j]+self.t[j]+1)**3

        val += x[i] + self.h*((1-self.t[i])*sum1 + self.t[i]*sum2)/2
       
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        i-=1

        for j in range(i):
            grd[j] += self.h*((1-self.t[i])*3*self.t[j]*(x[j]+self.t[j]+1)**2)/2
        for j in range(i,n):
            grd[j] += self.h*((1-self.t[j])*3*self.t[i]*(x[j]+self.t[j]+1)**2)/2
        grd[i] += 1.0
        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        i -= 1

        for j in range(i):
            hss[j,j] += self.h*((1-self.t[i])*6*self.t[j]*(x[j]+self.t[j]+1))/2
        for j in range(i,n):
            hss[j,j] += self.h*((1-self.t[j])*6*self.t[i]*(x[j]+self.t[j]+1))/2

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        i -= 1

        for j in range(i):
            trd[j,j,j] += self.h*((1-self.t[i])*6*self.t[j])/2
        for j in range(i,n):
            trd[j,j,j] += self.h*((1-self.t[j])*6*self.t[i])/2

        return trd

class BroydenTridiagonal(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10  
            x0 = np.ones(n) * (-1)
        super().__init__(x0, n)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = 0.0
        i-=1
        if i != 0 :
            val += -x[i-1]
        if i != n-1 :
            val += -2*x[i+1]
        
        val += 1 + (3 - 2*x[i])*x[i]
       
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        i-=1

        if i != 0 :
            grd[i-1] += -1
        if i != n-1 :
            grd[i+1] += -2
        
        grd[i] += (3 - 4*x[i])

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        i -= 1

        hss[i,i] += -4

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        for i in range(n):
            trd[i,i,i] = 96*x[i] - 72
            if i != 0 :
                trd[i-1,i,i] = 8
                trd[i,i-1,i] = 8
                trd[i,i,i-1] = 8
            if i != n-1 :
                trd[i+1,i,i] = 16
                trd[i,i+1,i] = 16
                trd[i,i,i+1] = 16

        return trd
#31--
class BroydenBanded(ChainRule):
    def __init__(self, n=None, x0=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n) * (-1)
        self.J = []
        for i in range(1, n + 1):
            lower_bound = max(1, i - 5)
            upper_bound = min(n, i + 1)
            indices = [j-1 for j in range(lower_bound, upper_bound + 1) if j != i]
            self.J.append(indices)
        super().__init__(x0, n)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = 0.0
        i-=1

        for j in self.J[i] :
            val += -x[j]*(1+x[j])
        val += (2 + 5*x[i]**2)*x[i] + 1
        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        i-=1

        grd[i] += 2 + 15*x[i]**2

        for j in self.J[i] :
            grd[j] += -(1 + 2*x[j])

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        i -= 1

        hss[i,i] += 30*x[i]

        for j in self.J[i] :
            hss[j,j] += -2

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])
        i-=1
        trd[i,i,i] = 30

        return trd

class LinearFullRank(ChainRule):
    def __init__(self, n=None, x0=None, m=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n)
        if m is None:
            m = n
        super().__init__(x0, m)

    def fi(self, x, i):
        n = np.shape(x)[0]
        val = 0.0
        i-=1
        sum = 0.0

        for j in range(n):
            sum += x[j]

        val += -2/self.m*sum - 1

        if i < n :
            val += x[i]

        return val        

    def gradi(self, x, i):
        n = np.shape(x)[0]
        grd = np.zeros([n])
        i-=1

        for j in range(n):
            grd[j] += -2/self.m

        if i < n :
            grd[i] += 1

        return grd
    
    def hessi(self, x, i):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])

        return hss
    
    def third_order_tensori(self, x, i):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        return trd

class LinearRank1(TestFunction):
    def __init__(self, n=None, x0=None, m=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n)
        if m is None:
            m = n
        super().__init__(x0, m)

    def f(self, x):
        n = np.shape(x)[0]
        val = 0.0
        sum = 0.0

        for j in range(n):
            sum += (j+1)*x[j]

        for i in range(self.m):
            val += (i+1)**2*sum**2 + 1 - 2*(i+1)*sum

        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        sum = 0.0

        for j in range(n):
            sum += (j+1)*x[j]

        for i in range(self.m):
            for j in range(n):
                grd[j] += 2*(i+1)**2*(j+1)*sum - 2*(i+1)*(j+1)


        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        for i in range(self.m):
            for j in range(n):
                for k in range(n):
                    hss[j,k] += 2*(i+1)**2*(j+1)*(k+1)

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        return trd

class LinearRank1ZeroColumnsAndRows(TestFunction):
    def __init__(self, n=None, x0=None, m=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 10
            x0 = np.ones(n)
        if m is None:
            m = n
        super().__init__(x0, m)

    def f(self, x):
        n = np.shape(x)[0]
        val = 2.0
        sum = 0.0

        for j in range(1,n-1):
            sum += (j+1)*x[j]

        for i in range(1,self.m-1):
            val += (i*sum - 1)**2

        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        sum = 0.0

        for j in range(1,n-1):
            sum += (j+1)*x[j]

        for i in range(1,self.m-1):
            for j in range(1,n-1):
                grd[j] += i*(sum*i - 1)*2*(j+1)


        return grd
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])
        
        sum = 0.0

        for j in range(1,n-1):
            sum += (j+1)*x[j]

        for i in range(1,self.m-1):
            for j in range(1,n-1):
                for k in range(1,n-1):
                    hss[k,j] += i**2*2*(j+1)*(k+1)

        return hss
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        return trd

class Chebyquad(TestFunction):
    def __init__(self, n=None, x0=None, m=None):
        if x0 is not None:
            x0 = np.asarray(x0)
            n = x0.shape[0]
        else:
            if n is None:
                n = 8
            x0 = np.array([j/(n+1) for j in range(1, n+1)])
        if m is None:
            m = n 

        self.integT = np.zeros(m+1)
        for i in range(m+1):
            if i % 2 == 1:
                self.integT[i] = 0.0
            else:
                if i == 0:
                    self.integT[i] = 1.0
                else:
                    self.integT[i] = -1.0 / (i*i - 1)

        super().__init__(x0,m)

    def all_cheby(self,x,n):
        chebys = [x, 2*x**2 - 1]
    
        if n == 0:
            return chebys[0].reshape(1, -1)
        if n == 1:
            return np.array(chebys[:1])
        
        for i in range(2, n):
            term = 2 * x * chebys[i-1] - chebys[i-2]
            chebys.append(term)
        
        return np.array(chebys[:n])
    
    def all_cheby_1st(self,x,n):
        chebys_1st = [np.ones_like(x), 4*x]

        chebys = self.all_cheby(x,n)
    
        if n == 0:
            return chebys,chebys_1st[0].reshape(1, -1)
        if n == 1:
            return chebys,np.array(chebys[:1])
        
        for i in range(2, n):
            term = 2 * x * chebys_1st[i-1] + 2 * chebys[i-1] - chebys_1st[i-2]
            chebys_1st.append(term)
        
        return chebys,np.array(chebys_1st[:n])
    
    def all_cheby_2nd(self,x,n):
        chebys_2nd = [np.zeros_like(x), 4*np.ones_like(x)]

        chebys,chebys_1st = self.all_cheby_1st(x,n)
    
        if n == 0:
            return chebys,chebys_1st,chebys_2nd[0].reshape(1, -1)
        if n == 1:
            return chebys,chebys_1st,np.array(chebys_2nd[:1])
        
        for i in range(2, n):
            term = 2 * x * chebys_2nd[i-1] + 4 * chebys_1st[i-1] - chebys_2nd[i-2]
            chebys_2nd.append(term)
        
        return chebys,chebys_1st,np.array(chebys_2nd[:n])
    
    def all_cheby_3rd(self,x,n):
        chebys_3rd = [np.zeros_like(x), np.zeros_like(x)]

        chebys,chebys_1st,chebys_2nd = self.all_cheby_2nd(x,n)
    
        if n == 0:
            return chebys,chebys_1st,chebys_2nd,chebys_3rd[0].reshape(1, -1)
        if n == 1:
            return chebys,chebys_1st,chebys_2nd,np.array(chebys_3rd[:1])
        
        for i in range(2, n):
            term = 2 * x * chebys_3rd[i-1] + 6 * chebys_2nd[i-1] - chebys_3rd[i-2]
            chebys_3rd.append(term)
        
        return chebys,chebys_1st,chebys_2nd,np.array(chebys_3rd[:n])
    
    def f(self, x):
        n = np.shape(x)[0]
        val = 0.0
        T = self.all_cheby(2*x-1,self.m)
        
        for i in range(self.m) :
            sum = 0.0
            for j in range(n) :
                sum += T[i,j]
            val += (sum/n - self.integT[i+1])**2

        return val        

    def grad(self, x):
        n = np.shape(x)[0]
        grd = np.zeros([n])

        T, dT = self.all_cheby_1st(2*x-1,self.m)

        for i in range(self.m) :
            sum = 0.0
            for j in range(n) :
                sum += T[i,j]
            for j in range(n) :
                grd[j] += 2*(sum/n - self.integT[i+1])*dT[i,j]/n

        return grd*2
    
    def hess(self, x):
        n = np.shape(x)[0]
        hss = np.zeros([n,n])

        T, dT, ddT = self.all_cheby_2nd(2*x-1,self.m)

        for i in range(self.m) :
            sum = 0.0
            for j in range(n) :
                sum += T[i,j]
            for j in range(n) :
                hss[j,j] += 2*(sum/n - self.integT[i+1])*ddT[i,j]/n
                for k in range(n):
                    hss[j,k] += 2*dT[i,j]*dT[i,k]/n**2

        return hss*4
    
    def third_order_tensor(self, x):
        n = np.shape(x)[0]
        trd = np.zeros([n,n,n])

        T, dT, ddT, dddT = self.all_cheby_3rd(2*x-1,self.m)

        for i in range(self.m) :
            sum = 0.0
            for j in range(n) :
                sum += T[i,j]
            for j in range(n) :
                trd[j,j,j] += 4*dT[i,j]*ddT[i,j]/n**2 +  2*(sum/n - self.integT[i+1])*dddT[i,j]/n - 2*dT[i,j]*ddT[i,j]/n**2
                for k in range(n):
                    trd[j,j,k] += 2*dT[i,k]*ddT[i,j]/n**2
                    trd[j,k,k] += 2*dT[i,j]*ddT[i,k]/n**2

        return trd*8

