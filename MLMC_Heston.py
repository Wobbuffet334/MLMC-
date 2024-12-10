# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:47:50 2024

@author: hvermeer
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as ss
from scipy.integrate import quad
from functools import partial

my_generator = np.random.default_rng()        

class HestonModel:
    def __init__(self, initial_index, initial_variance, nominal, real, mean_reversion_variance, correlation, mean_variance, volatility_of_variance, T):
        """  Specificies the (initial) parameters of the Heston model. These are independent of the used simulation method.  
        
        Parameters
        -----------
        initial_index : Initial value of the inflation index.
        initial_variance : Initial value of the variance process.
        nominal : Constant value of the nominal interest rate.
        real : Constant value of the real interest rate.
        mean_reversion_variance : Mean reversion coefficient of the variance process (alpha).
        correlation : Correlation coefficient between the brownian motions of the inflation index and variance process. 
        mean_variance : Average level of the variance process.
        volatility_of_variance : Volatility parameter of the variance process.
        T : Time to maturity
        
        """    
        self.I0 = initial_index
        self.V0 = initial_variance
        self.RN = nominal
        self.RR = real
        self.alpha = mean_reversion_variance
        self.rho = correlation
        self.Vbar = mean_variance
        self.sigmaV = volatility_of_variance
        self.T = T       
        
    def correlated_BM(self, NoOfPaths, NoOfSteps):
        """  Constructs a 2-dimensional array of correlated standard normal variables using Cholesky decomposition. One standard normal variable per time step per sample path. 
        
        Parameters
        -----------
        NoOfPaths: Number of samples 
        NoOfSteps: Number of time steps
        
        Returns: Array of correlated standard normal variables.
        """

        rho = self.rho

        C = np.array([[1, rho], [rho, 1]])
        
        L = np.linalg.cholesky(C)
        
        W = np.random.normal(0, 1, (2, NoOfPaths, NoOfSteps))
            
        CW = np.einsum('ij,jkl->ikl', L, W)

        return CW
    
    
    def cf_heston(self, u, t):
        """  The characteristic function of the Heston (1993) model 
        
        
        """

        xi = self.alpha - self.sigmaV * self.rho * u * 1j
        
        mu = self.RN - self.RR
        
        d = np.sqrt(xi**2 + self.sigmaV**2 * (u**2 + 1j * u))

        g = (xi - d) / (xi + d)

        cf = np.exp(1j * u * mu * t + (self.alpha * self.Vbar) / (self.sigmaV ** 2) * ((xi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
                    + (self.V0 / self.sigmaV ** 2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
        
        return cf
    
    def cf_heston_partial(self, T):
        
        cf_H_b = partial(self.cf_heston, t=T)
        
        return cf_H_b
    
    def Q1(self, k, cf, right_lim):
        """Numerical integration of the characteristic fucntion using scipy.integrate. Probability to be in the money under the stock numeraire.
        
        Parameters
        -----------
        cf : characteristic function 
        right_lim : right limit of integration
        
        """
       
        def integrand(u):
            return np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))
        
        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]

    def Q2(self, k, cf, right_lim):
        """ Probability to be in the money under the money market numeraire
        
        Parameters
        -----------
        
        cf: characteristic function
        right_lim: right limit of integration
        
        """

        def integrand(u):
            return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))
        
        return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]
    
    def OptionValueCos(self, K, N, a, b, T):
        """  Calculates the analytical value of a call option on the Heston model using its characteristic function  
        
        Parameters
        -----------
        K : Strike prices
        N : Number of Fourier terms
        a : Left limit
        b : Right limit
        T : Time to maturity
        """
        
        bma = b-a
        q = 0 # dividend yield
        k = np.arange(0, N + 1) 

        u = k *np.pi/(b-a)
        V_CALL = np.zeros((np.size(K)))
        V_PUT = np.zeros((np.size(K)))
        V_PCP = np.zeros((np.size(K)))
        cf = self.cf_heston(u, T)

        
        # Iterate through the strike prices.
        for m in range(0, np.size(K)):
            x = np.log(self.I0/K[m])
            Term = np.exp(1j * k * np.pi * (x-a) / bma)
            Fk = np.real(cf * Term)
            Fk[0] = 0.5 * Fk[0]
            V_CALL[m] = K[m] * np.sum(Fk * Uk(k, a, b, 'call')) * np.exp(-(self.RN - self.RR) * T)
            V_PUT[m] = K[m] * np.sum(Fk * Uk(k, a, b, 'put')) * np.exp(-(self.RN - self.RR) * T)
            V_PCP[m] = V_PUT[m] + self.I0 * np.exp(-q * T) - K[m] * np.exp(-(self.RN - self.RR) * T)
            
        return V_PCP, cf
 
    def euler_step(self, index, v, z1, z2, dt):
        """   Performs one iteration of the euler discretization step for the log heston model  
        
        Parameters
        -----------
        index : current value of the inflation index
        v : current value of the variance process
        z1 : correlated sample from standard normal distribution for the inflation index
        z2 : correlated sample from standard normal distribution for the variance process
        dt : time step 
            
        """
        
        r = self.RN - self.RR
        v_sq = np.sqrt(np.maximum(v,0))
        dt_sq = np.sqrt(dt)
        
        index = index * np.exp((r - 0.5 * v) * dt + v_sq * dt_sq * z1)
        
        v = v + self.alpha * (self.Vbar - v) * dt + self.sigmaV * v_sq * dt_sq * z2

        return index, v
    
    def euler_step_truncated(self, index, v, z1, z2, dt, Wdt):
        """   Performs one iteration of the full truncated euler discretization step for the log heston model 
        
        Parameters
        -----------
        index : current value of the inflation index
        v : current value of the variance process
        z1 : correlated sample from standard normal distribution for the inflation index
        z2 : correlated sample from standard normal distribution for the variance process
        dt : time step 
        Wdt : time step used for the brownian motion
        
        """
       
        r = self.RN - self.RR
        v_sq = np.sqrt(np.maximum(v,0))
        Wdt_sq = np.sqrt(Wdt)
        
        index = index * np.exp((r - 0.5 * np.maximum(v,0)) * dt + v_sq * Wdt_sq * z1)
        
        v = v + self.alpha * (self.Vbar - np.maximum(v, 0)) * dt + self.sigmaV * v_sq * Wdt_sq * z2

        return index, v
    
    def precomputations_NCI(self):
        """  Precomputes the grid of inverse chi squared distribution values used for the NCI simulation scheme.
        
        Parameters
        -----------
        Only requires Heston model parameters.
        
        """
        d = 4 * self.Vbar * self.alpha / self.sigmaV ** 2
        
        Qmax = 40 # Adjust the grid sizes here
        Q = np.linspace(0, Qmax, Qmax + 1)
        
        delta = 10**(-15)
        U_grid = np.linspace(0, 1-delta, 10001)
        
        inverses = np.zeros((len(Q), len(U_grid)))
        
        for q in Q:
            q = int(q)
            inverses[q] = ss.chi2.ppf(U_grid, d + 2*q)
        
        U_distance = U_grid[1] - U_grid[0]
        m = np.zeros((len(Q), len(U_grid)))
        
        m[:, 0] = (inverses[:, 1] - inverses[:, 0]) / U_distance
        m[:, -1] = (inverses[:, -1] - inverses[:, -2]) / U_distance
        
        k = 1
        while k < len(U_grid) - 1:

            m[:, k] = (inverses[:, k + 1] - inverses[:, k - 1]) / (2 * U_distance)
            k += 1
        
        return inverses, m
    
    def NCI_method_linear_interpol(self, v, inverses, dt):
        """ Samples the variance process for the nexts timestep using the NCI method and linear interpolation
        
        Parameters
        -----------
        
        v : current value of the variance process
        inverses : precomputed grid of inverse values
        dt : timestep
        
        """
        d = 4 * self.Vbar * self.alpha / self.sigmaV ** 2
        lam = v * 4 * self.alpha * np.exp(-self.alpha * dt) / (self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)))
        C0 = self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)) / (4 * self.alpha)
        
        Qmax = 40
        
        delta = 10**(-15)
        U_grid = np.linspace(0, 1-delta, 10001)
          
        Q_sample = np.random.poisson(lam / 2)
        U_sample = np.random.uniform(0, 1)

        
        index = int(np.floor(U_sample * (len(U_grid) - 1)))
      
        if Q_sample < Qmax:
            Upper = (U_grid[index + 1] - U_sample) / (U_grid[index + 1] - U_grid[index])
            Lower = (U_sample - U_grid[index]) / (U_grid[index + 1] - U_grid[index])
            v = C0 * (Upper * inverses[Q_sample, index] + Lower * inverses[Q_sample, index + 1])

            
        else:
            ncxd = my_generator.noncentral_chisquare(d, lam)
            v = C0 * ncxd

        return v
    
    def NCI_method_cubic_hermite_interpol(self, v, inverses, dt, m):
        """ Samples the variance process for the next timestep using the NCI method and linear interpolation
    
        Parameters
        -----------
        
        v : current value of the variance process
        inverses : precomputed grid of inverse values
        dt : timestep
        
        NOTE: This is still a work in progress. It is currently not optimised for path-dependent simulation.
        
        """
        d = 4 * self.Vbar * self.alpha / self.sigmaV ** 2
        lam = v * 4 * self.alpha * np.exp(-self.alpha * dt) / (self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)))
        C0 = self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)) / (4 * self.alpha)

        Qmax = 20
        
        delta = 10**(-15)
        U_grid = np.linspace(0, 1-delta, 10001)
          

        Q_sample = np.random.poisson(lam / 2)
        U_sample = np.random.uniform(0, 1)

        
        index = int(np.floor(U_sample * (len(U_grid) - 1)))
        t = (U_sample - U_grid[index]) / (U_grid[1] - U_grid[0])

        if Q_sample < Qmax:
            h_00 = 2 * t**3 - 3 * t**2 + 1.0
            h_10 = t**3 - 2 * t**2 + t
            h_01 = -2 * t**3 - 3 * t**2
            h_11 = t**3 - t**2
            triangle = (inverses[Q_sample, index] - inverses[Q_sample, index + 1]) / (U_grid[1] - U_grid[0])
            
            J_UV = h_00 * inverses[Q_sample, index] + h_01 * inverses[Q_sample, index + 1] + triangle * (m[Q_sample, index] * h_10 + m[Q_sample, index] * h_11)
            v = C0 * J_UV
  
        
        else:
            ncxd = my_generator.noncentral_chisquare(d, lam)
            v = C0 * ncxd
        
        return v

    def quadratic_exponential(self, v, dt):
        """ Samples the variance process for the next timestep using the QE method.
        
        Parameters
        -----------
        
        v : current value of the variance process
        dt : timestep
        
        """
        
        C1 = self.sigmaV**2 * np.exp(-self.alpha * dt) * (1 - np.exp(-self.alpha * dt)) / self.alpha
        C2 = self.Vbar * self.sigmaV**2 * (1 - np.exp(-self.alpha * dt))**2 / (2 * self.alpha)
        C3 = np.exp(-self.alpha * dt)
        C4 = self.Vbar * (1 - np.exp(-self.alpha * dt))
        
        s2 = C1 * v + C2
        m = C3 * v + C4
        
        psi = s2 / (m**2)
        psi_c = 1.5

        if psi <= psi_c:
            b = np.sqrt(2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(2 / psi - 1))
            a = m / (1 + b**2)
            Z = np.random.normal(0,1)
            V = a * (b + Z)**2
            
        elif psi > psi_c:
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            u = np.random.uniform(0,1)
            if u <= p:
                V = 0
            elif u > p:
                V = (1 / beta) * np.log((1 - p) / (1-u))
        
        return V
        
        
    def exact_trapezoidal_step_fine(self, index, v, z1, dt):
        """  Performs one iteration of the exact simulation method. The variance is sampled using the built in generator package.
        For the integrated variance process the trapezoidal scheme is used. 
        
        Parameters
        -----------
        index : current value of the inflation index
        v : current value of the variance process
        z1 : Sample from standard normal distribution for the inflation index
        dt : time step 
            
        """
        
        r = self.RN - self.RR
        
        d = 4 * self.Vbar * self.alpha / self.sigmaV ** 2
        lam = v * 4 * self.alpha * np.exp(-self.alpha * dt) / (self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)))
        
        ncxd = my_generator.noncentral_chisquare(d, lam)
        
        v_t = ncxd * self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)) / (4 * self.alpha) 
        
        v_trap = (v_t + v) * dt / 2
        
    
        term1 = (r - self.rho * self.alpha * self.Vbar / self.sigmaV) * dt
        term2 = (self.rho * self.alpha /self.sigmaV - 1/2) * v_trap
        term3 = self.rho * (v_t - v) / self.sigmaV
        term4 = np.sqrt(1 - self.rho ** 2) * np.sqrt(v_trap) * z1
        
        index = index * np.exp(term1 + term2 + term3 + term4)
        
        return index, v_t
    
    
    def exact_trapezoidal_step_coarse(self, index, v1, v2, z1, dt):
        """  Performs one iteration of the exact simulation method specifically for the coarse level used in multilevel Monte Carlo.
        
        Parameters
        -----------
        index : current value of the inflation index
        v1 : current value of the variance process
        v2 : sampled future value of the variance process
        z1 : Sample from standard normal distribution for the inflation index
        dt : time step 
            
        """
        
        v_trap = (v1 + v2) * dt / 2
        
        term1 = (r - self.rho * self.alpha * self.Vbar / self.sigmaV) * dt
        term2 = (self.rho * self.alpha /self.sigmaV - 1/2) * v_trap
        term3 = self.rho * (v2 - v1) / self.sigmaV
        term4 = np.sqrt(1 - self.rho ** 2) * np.sqrt(v_trap) * z1
        
        index = index * np.exp(term1 + term2 + term3 + term4)
        
        return index
            
    def simulate_paths_euler(self, NoOfSteps, NoOfPaths):
        """ Simulate the complete paths of the inflation index and variance process using the Euler discretization
        
        Parameters
        -----------
        NoOfSteps: Number of time steps
        NoOfPaths: Number of samples
        
        """
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        dt = self.T / NoOfSteps
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        W_V = correlated_normal[1, :, :]

        for i in range(1, NoOfSteps + 1):
            new_inf, new_var = self.euler_step(inflation[:, i-1], variance[:,i-1], W_I[:, i-1], W_V[:, i-1], dt)
            inflation[:, i] = new_inf
            variance[:, i] = new_var
        
        
        return inflation, variance
    
    def simulate_paths_eulerFT(self, NoOfSteps, NoOfPaths):
        """ Simulate the complete paths of the inflation index and variance process using the full truncated Euler discretization
        
        Parameters
        -----------
        NoOfSteps: Number of time steps
        NoOfPaths: Number of samples
        
        """
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        dt = self.T / NoOfSteps
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        W_V = correlated_normal[1, :, :]

        for i in range(1, NoOfSteps + 1):
            new_inf, new_var = self.euler_step_truncated(inflation[:, i-1], variance[:,i-1], W_I[:, i-1], W_V[:, i-1], dt, dt)
            inflation[:, i] = new_inf
            variance[:, i] = new_var
        
        
        return inflation, variance
    
    def simulate_paths_exact_trap(self, NoOfSteps, NoOfPaths):
        """ Simulate the complete paths of the inflation index and variance process using the exact simulation scheme where the 
        integrated variance is approximated with the trapezoidal scheme.
        
        Parameters
        -----------
        NoOfSteps: Number of time steps
        NoOfPaths: Number of samples
        
        """
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        dt = self.T / NoOfSteps
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]

        for i in range(1, NoOfSteps + 1):
            new_inf, new_var = self.exact_trapezoidal_step_fine(inflation[:, i-1], variance[:,i-1], W_I[:, i-1], dt)
            inflation[:, i] = new_inf
            variance[:, i] = new_var
        
        
        return inflation, variance
    
    def simulate_paths_exact_NCI(self, NoOfSteps, NoOfPaths):
        """ Simulate the complete paths of the inflation index and variance process using the NCI simulation scheme.
        
        Parameters
        -----------
        NoOfSteps: Number of time steps
        NoOfPaths: Number of samples
        
        """
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        dt = self.T / NoOfSteps
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        inverses, m = self.precomputations_NCI()
        
        for i in range(1, NoOfSteps + 1):
            for j in range(0, NoOfPaths):
                new_var = self.NCI_method_linear_interpol(variance[j, i-1], inverses, dt)
                variance[j, i] = new_var
        
            new_inf = self.exact_trapezoidal_step_coarse(inflation[:, i-1], variance[:, i-1], variance[:, i],  W_I[:, i-1], dt)
            inflation[:, i] = new_inf
        
        
        return inflation, variance
    
    def simulate_paths_exact_QE(self, NoOfSteps, NoOfPaths):
        """ Simulate the complete paths of the inflation index and variance process using the QE simulation scheme.
        
        Parameters
        -----------
        NoOfSteps: Number of time steps
        NoOfPaths: Number of samples
        
        """
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        dt = self.T / NoOfSteps
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]

        for i in range(1, NoOfSteps + 1):
            for j in range(0, NoOfPaths):
                new_var = self.quadratic_exponential(variance[j, i-1],  dt)
                variance[j, i] = new_var
        
            new_inf = self.exact_trapezoidal_step_coarse(inflation[:, i-1], variance[:, i-1], variance[:, i],  W_I[:, i-1], dt)
            inflation[:, i] = new_inf

        return inflation, variance
 
    
    def simulate_paths_exact_trapezoidal_MLMC(self, NoOfPaths, M, l):
        """ Simulate the complete paths of the inflation and variance process at the coarse and fine levels using exact simulation for
        the variance process. Used specifically for multilevel Monte Carlo.
        
        Parameters
        -----------
        NoOfPaths: Number of samples
        M : Difference in size of the time step between fine level and coarse level
        l : current level
        
        """
        
        NoOfSteps = M**l

        dt = self.T / NoOfSteps
        
        
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0

        
        if l == 0:
            inflation_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            variance_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
        else:
            inflation_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            variance_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            
        inflation_prev[:, 0] = self.I0
        variance_prev[:, 0] = self.V0
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        
        if l == 0:
            for i in range(1, NoOfSteps + 1):
                new_inf, new_var = self.exact_trapezoidal_step_fine(inflation[:, i-1], variance[:, i-1], W_I[:, i-1], dt)
                inflation[:, i] = new_inf
                variance[:, i] = new_var
                
            return inflation, variance, inflation_prev, variance_prev
            
        else:
            for i in range(1, int(NoOfSteps/M) + 1):
                W_I_prev = np.zeros((NoOfPaths))
                
                
                for m in range(0, M):
                    new_inf, new_var = self.exact_trapezoidal_step_fine(inflation[:, (i-1)*M + m], variance[:, (i-1)*M + m], W_I[:, (i-1)*M + m], dt)
                    inflation[:, (i-1)*M + m + 1] = new_inf
                    variance[:, (i-1)*M + m + 1] = new_var

                    W_I_prev += W_I[:, (i-1)*M + m]
                
                W_I_prev = W_I_prev / np.sqrt(M)


                new_inf_prev = self.exact_trapezoidal_step_coarse(inflation_prev[:, i-1], variance[:, (i-1) * M], variance[:, i * M],  W_I_prev, M * dt)
                
                inflation_prev[:, i] = new_inf_prev

            return inflation, variance, inflation_prev, variance_prev
    
    
    def simulate_paths_euler_MLMC(self, NoOfPaths, M, l, h_0):
        """ Simulate the complete paths of the inflation and variance process at the coarse and fine levels using the Euler scheme. 
        Used specifically for multilevel Monte Carlo.
        
        Parameters
        -----------
        NoOfPaths: number of samples.
        M : difference in size of the time step between fine level and coarse level.
        l : current level.
        h_0 : adjust the size of the timestep in the first level.
        
        """
        
        NoOfSteps = h_0 * M**l 
        
        dt = self.T / NoOfSteps
        
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0

        
        if l == 0:
            inflation_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            variance_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
        else:
            inflation_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            variance_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            
        inflation_prev[:, 0] = self.I0
        variance_prev[:, 0] = self.V0
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        W_V = correlated_normal[1, :, :]
        
        if l == 0:
            for i in range(1, NoOfSteps + 1):
                new_inf, new_var = self.euler_step_truncated(inflation[:, i-1], variance[:, i-1], W_I[:, i-1], W_V[:, i-1], dt, dt)
                inflation[:, i] = new_inf
                variance[:, i] = new_var
                
            return inflation, variance, inflation_prev, variance_prev
            
        else:
            for i in range(1, int(NoOfSteps/M) + 1):
                W_I_prev = np.zeros((NoOfPaths))
                W_V_prev = np.zeros((NoOfPaths))
                
                for m in range(0, M):
                    new_inf, new_var = self.euler_step_truncated(inflation[:, (i-1)*M + m], variance[:, (i-1)*M + m], W_I[:, (i-1)*M + m], W_V[:, (i-1)*M + m], dt, dt)
                    inflation[:, (i-1)*M + m + 1] = new_inf
                    variance[:, (i-1)*M + m + 1] = new_var

                    W_I_prev += W_I[:, (i-1)*M + m]
                    W_V_prev += W_V[:, (i-1)*M + m]

                new_inf_prev, new_var_prev = self.euler_step_truncated(inflation_prev[:, i-1], variance_prev[:, i-1], W_I_prev, W_V_prev, M * dt, dt)
                
                inflation_prev[:, i] = new_inf_prev
                variance_prev[:, i] = new_var_prev
            
    
            return inflation, variance, inflation_prev, variance_prev
    
    
    def prices_MLMC(self, K, inflation, inflation_prev):
        """ Calculates the value of the payoff function at the coarse level and the fine level.
        
        Parameters
        -----------
        K : strike price
        inflation : inflation index paths at the fine level
        inflation_prev : inflation index paths at the coarse level
        
        """
        
        r = self.RN - self.RR
        
        
        P_f = np.exp(-r * self.T) * np.maximum(inflation[:,-1] - K, 0)
        P_c = np.exp(-r * self.T) * np.maximum(inflation_prev[:,-1] - K, 0)
        
        return P_f, P_c
        
    
    def MLMC(self, K, error):
        """ Performs the multilevel Monte Carlo algorithm using the exact simulation method. Returns the computed price, the values
        of the estimators and their variances, the number of levels used, and the results using Monte Carlo simulation without MLMC.
        
        Parameters
        -----------
        K : strike price
        error : target mean square error 
        
        """
        
        L = 2 # Starting number of levels.
        N_0 = 10000 # Initial number of samples.
        
        Nl = np.zeros(3) # Initialise the array containing the optimal number of samples.
        dNl = np.array([N_0, N_0, N_0]) # Initialise the array containing the difference between the optimal number of samples and already computed number of samples.
        
        
        # Stepsize parameters.
        h_0 = 1 
        M = 4
        
        # Convergence parameters.
        xi = error #Error goal
        beta = 2
        alpha = 2
        gamma = 2
        
        # Initialise the sums used for estimating.
        sum1 = np.zeros(3)
        sum2 = np.zeros(3)
        sum3 = np.zeros(3)
        sum4 = np.zeros(3)
        
        # While additional samples need to be calculated, keep iterating.
        while sum(dNl) > 0:
            for l in range(0, L+1):
                if dNl[l] > 0:
                    if dNl[l]*M**l > 10**8: #Helps with memory issues
                        dNl[l] = 10**8/(M**l)
                    inflation_paths, variance_paths, inflation_prev, variance_prev = self.simulate_paths_exact_trapezoidal_MLMC(int(dNl[l]), M, l)
                    # inflation_paths, variance_paths, inflation_prev, variance_prev = self.simulate_paths_euler_MLMC(int(dNl[l]), M, l, h_0)

                    P_f, P_c = self.prices_MLMC(K, inflation_paths, inflation_prev)
                    
                    Nl[l] += dNl[l]
                    
                    sums1 = np.sum(P_f - P_c)
                    sums2 = np.sum((P_f - P_c)**2)
                    sums3 = np.sum(P_f)
                    sums4 = np.sum(P_f**2)
                    
                    sum1[l] += sums1
                    sum2[l] += sums2
                    sum3[l] += sums3
                    sum4[l] += sums4
        
            m_l = np.abs(sum1 / Nl)
            v_l = np.maximum(0, sum2 / Nl -  m_l**2)

            standardMC_mean = np.abs(sum3 / Nl)
            standardMC_var = sum4 / Nl -  standardMC_mean**2
            
            # Estimate optimal samples.
            
            h_l = np.array([self.T / (M**l) for l in range(0, L+1)])
 
            N_s = np.ceil(2 * np.sqrt(v_l * h_l) * np.sum(np.sqrt(v_l / h_l)) / xi**2)
            
            dNl = np.maximum(0, N_s - Nl)

            
            # Estimate remaining error and decide whether a new level is required.
            
            if np.sum( dNl > 0.01*Nl ) == 0:
                rem_error = np.array([m_l[-3]  / (2 ** (-2 * alpha)), m_l[-2]  / (2 ** (-alpha)), m_l[-1]]) / (M**alpha - 1)
               
                rem_error = np.max(rem_error)

                if rem_error > xi/np.sqrt(2):
                    L += 1

                    v_l = np.append(v_l, v_l[L-1] / M**beta )
                    Nl = np.append(Nl, 0)
                    

                    h_l = np.array([self.T / (M**l) for l in range(0, L+1)])

                    N_s = np.ceil(2 * np.sqrt(v_l * h_l) * np.sum(np.sqrt(v_l / h_l)) / xi**2)

                    sum1 = np.append(sum1, 0)
                    sum2 = np.append(sum2, 0)
                    sum3 = np.append(sum3, 0)
                    sum4 = np.append(sum4, 0)

                    dNl = np.maximum(0, N_s-Nl)
                
        # Evaluate the MLMC estimator.
        
        MLMC_price = np.sum(sum1 / Nl)
        
        return MLMC_price, m_l, v_l, Nl, L, standardMC_mean, standardMC_var
    
    def plot_paths(self, inflation_paths, variance_paths, time_values):
        """ Plots the paths of the inflation index and the variance 
        
        """
        # Visualize the simulation results
        plt.figure(figsize=(12, 8))

        for i in range(inflation_paths.shape[0]):
            plt.plot(time_values, inflation_paths[i], label=f'Inflation Path {i+1}')
            
        plt.title('Simulated Inflation Paths')
        plt.xlabel('Time')
        plt.ylabel('Inflation')
        plt.show()
        
        plt.figure(figsize=(12, 8))

        for i in range(variance_paths.shape[0]):
            plt.plot(time_values, variance_paths[i], label=f'Variance Path {i+1}')

        plt.title('Simulated Variance Paths')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.show()
        
    def plot_means(self, base, m_l, m_MC, L):
        """ Plots the values of the estimators used in multilevel Monte Carlo. Log base M of the estimator is used to show convergence.
        
        Parameters
        -----------
        base : value for M used
        m_l : the values of the MLMC estimators for each level
        m_MC : the values of the Monte Carlo estimator for each level
        L : the maximum level reached
        
        """
        
        levels = np.linspace(0, L, L+1)
        
        log_means_MLMC = np.emath.logn(base, m_l)
        log_means_MC = np.emath.logn(base, m_MC)
        
        plt.figure()
        plt.plot(levels, log_means_MC, marker = '*', label = r'$E(\hat{P}_{\ell})$',linestyle = '--')
        plt.plot(levels, log_means_MLMC, marker = '*', label = r'$E(\hat{P}_{\ell} -\hat{P}_{\ell-1})$',linestyle = '--')
        plt.title('Expectation MLMC vs MC')
        plt.ylabel(r'$\log_{4}(|mean|)$')
        plt.xlabel(r'level ($\ell$)')
        plt.legend()
        plt.show()
       
    def plot_variances(self, base, v_l, Var_MC, L):
        """ Plots the variances of the estimators used in multilevel Monte Carlo. Log base M of the estimator is used to show convergence.
        
        Parameters
        -----------
        base : value for M used
        v_l : the variances of the MLMC estimators for each level
        Var_MC : the variances of the Monte Carlo estimator for each level
        L : the maximum level reached
        
        """
        levels = np.linspace(0, L, L+1)
        
        log_variances_MLMC = np.emath.logn(4, v_l)
        log_variances_MC = np.emath.logn(4, Var_MC)
        
        plt.figure()
        plt.plot(levels, log_variances_MC, marker = '*', label = r'$Var(\hat{P}_{\ell})$', linestyle = '--')
        plt.plot(levels, log_variances_MLMC, marker = '*', label = r'$Var(\hat{P}_{\ell} -\hat{P}_{\ell-1})$',linestyle = '--')
        plt.title('Variance MLMC vs MC')
        plt.ylabel(r'$\log_{4}(variance)$')
        plt.xlabel(r'level ($\ell$)')
        plt.legend()
        plt.show()
        
    def truncationRange(self, L, T):
        """ Calculated the parameters a and b specified for the COS method. 
        
        Parameters
        -----------
        L = truncation range
        T = time to maturity
        
        """
        mu = self.RN-self.RR
        sigma = self.V0
        lm = self.alpha
        v_bar = self.Vbar
        rho = self.rho
        volvol = self.sigmaV
        
        c1 = mu * T + (1 - np.exp(-lm *T)) * (v_bar - sigma)/(2 * lm) - v_bar * T / 2

        c2 = 1/(8 * np.power(lm,3)) * (volvol * T * lm * np.exp(-lm * T) \
            * (sigma - v_bar) * (8 * lm * rho - 4 * volvol) \
            + lm * rho * volvol * (1 - np.exp(-lm * T)) * (16 * v_bar - 8 * sigma) \
            + 2 * v_bar * lm * T * (-4 * lm * rho * volvol + np.power(volvol,2) + 4 * np.power(lm,2)) \
            + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2 * lm * T) \
            + v_bar * (6 * np.exp(-lm * T) - 7) + 2 * sigma) \
            + 8 * np.power(lm,2) * (sigma - v_bar) * (1 - np.exp(-lm * T)))

        a = c1 - L * np.sqrt(np.abs(c2))
        b = c1 + L * np.sqrt(np.abs(c2))
        return a, b
    
def Chi(k, a, b, c, d):
    bma = b-a
    uu  = k * np.pi/bma
    
    chi = (1 / (1 + uu**2)) * (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + uu * np.sin(uu * (d-a)) * np.exp(d) - uu * np.sin(uu * (c-a)) * np.exp(c))

    return chi

def Psi(k, a, b, c, d):
    bma = b-a
    uu = k * np.pi / bma
    uu[0] = 1
    
    psi = (np.sin(uu * (d-a)) - np.sin(uu * (c-a))) / uu
    psi[0] = d-c

    return psi
    
def Uk(k, a, b, type):
    bma = b-a
    if type == 'put':
        Uk  = (2 / bma) * (-Chi(k,a,b,a,0) + Psi(k,a,b,a,0))
    elif type == 'call':
        Uk = (2 / bma) * (Chi(k,a,b,0,b) - Psi(k,a,b,0,b))
    return Uk 


"""""""  Specify the parameter values """""""

initial_index = 100.0 # I(0)
nominal = 0.06 # Nominal interest rate
real = 0.06 # Real interest rate
r = nominal - real


correlation = -0.9 # Correlation between the inflation index and the variance process

mean_variance = 0.04 # V_bar
volatility_of_variance = 1.0 # sigma_V
initial_variance = 0.04 # V(0)
mean_reversion_variance = 0.5 # alpha

T = 10.0 # Time to maturity

K = 100.0




print('Feller condition is satisfied if > 1 :', 2*mean_variance*mean_reversion_variance / volatility_of_variance**2)

Heston_Model = HestonModel(initial_index, initial_variance, nominal, real, mean_reversion_variance, correlation, mean_variance, volatility_of_variance, T)




# inverses, m = Heston_Model.precomputations_NCI() 


# v1 = []
# start = time.time()
# for i in range(1, 100000):
#     v_new = Heston_Model.NCI_method_linear_interpol(initial_variance, inverses, 0.0390625)
#     v1.append(v_new)
# end = time.time()
# print('time elapsed =', end - start)
# v2 = []    

# start = time.time()
# for i in range(1, 100000):
#     v_new = Heston_Model.exact_trapezoidal_step_fine2(initial_variance, 0.0390625)
#     v2.append(v_new)
# end = time.time()
# print('time elapsed =', end - start)
# start = time.time()

# v3=[]
# for i in range(1, 100000):
#     v_new = Heston_Model.NCI_method_cubic_hermite_interpol(initial_variance, inverses, 0.0390625, m)
#     v3.append(v_new)
# end = time.time()
# print('time elapsed =', end - start)

# v4=[]
# for i in range(1, 100000):
#     v_new = Heston_Model.quadratic_exponential(initial_variance, 0.0390625)
#     v4.append(v_new)
# end = time.time()
# print('time elapsed =', end - start)

# plt.figure()
# plt.hist(v1, bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
# plt.show()

# plt.figure()
# plt.hist(v2, bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
# plt.show()

# plt.figure()
# plt.hist(v3, bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
# plt.show()  

# plt.figure()
# plt.hist(v4, bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
# plt.show()  
 







""""""""""  Numerical integration with scipy integration package  """""""""""

limit_max = 1000  # right limit in the integration
# u = np.linspace(-4, 4, 100)

st = time.time()
cf_H_b = Heston_Model.cf_heston_partial(T)

limit_max = 1000

k = np.log(K / initial_index)
call = initial_index * Heston_Model.Q1(k, cf_H_b, limit_max) - 100 * np.exp(-r * T) * Heston_Model.Q2(k, cf_H_b, limit_max)
et = time.time()
print("Heston numerical integration call price: ", call)



"""""""""  Calculate the exact value using the COS Method  """""""""


L = 120
a,b = Heston_Model.truncationRange(L, T)

K = [100]
q=0
N = 2**15


OptionValuewithCos, cf = Heston_Model.OptionValueCos(K, N, a, b, T)

print('Option value using COS method', OptionValuewithCos)


N = np.linspace(300, 10000, 200)

Values = []
for i in N:
    OptionValuewithCos, cf = Heston_Model.OptionValueCos(K, i, a, b, T)
    Values.append(OptionValuewithCos[0])
Calls= [OptionValuewithCos for i in enumerate(N)]

plt.figure()
plt.plot(N, Values)
plt.plot(N, Calls)
plt.xlabel('Number of Fourier Terms')
plt.ylabel('Estimated price')
plt.show()

OptionValuewithCos, cf = Heston_Model.OptionValueCos(K, 2**15, a, b, T)

""""" Standard Monte Carlo simulation using different simulation schemes """""


K = 100
NoOfPaths = 100000

for i in range(0, 7):
    dt = T / 4**i
    NoOfSteps = 4**i
    print(NoOfSteps)
    st = time.time()
    P = np.array([])
    for i in range(0, 10):
        inflation_paths, variance_paths = Heston_Model.simulate_paths_euler(NoOfSteps, NoOfPaths) 
        P = np.append(P, np.exp(-r * T) * np.maximum(inflation_paths[:,-1] - K, 0))
        
    et = time.time()
    MC_mean = np.mean(P)
    error = np.sqrt(np.var(P) / (NoOfPaths*10))
    bias = OptionValuewithCos - MC_mean
    print('For dt = ', dt, 'the monte carlo price with Euler is', MC_mean, 'and the standard error is', error,' and bias', bias,'. This took', et - st,'seconds')
    

for i in range(0, 7):
    dt = T / 4**i
    NoOfSteps = 4**i
    st = time.time()
    DiscountedPayoff = 0
    P = np.array([])
    for i in range(0, 10):
        inflation_paths, variance_paths = Heston_Model.simulate_paths_euler(NoOfSteps, NoOfPaths) 
        P = np.append(P, np.exp(-r * T) * np.maximum(inflation_paths[:,-1] - K, 0))
        
    et = time.time()
    MC_mean = np.mean(P)
    error = np.sqrt(np.var(P) / (NoOfPaths*10))
    bias = OptionValuewithCos - MC_mean
    print('For dt = ', dt, 'the monte carlo price with full truncated Euler is', MC_mean, 'and the standard error is', error,' and bias', bias,'. This took', et - st,'seconds')
    


for i in range(0, 7):
    dt = T / 4**i
    NoOfSteps = 4**i
    st = time.time()
    P = np.array([])
    for i in range(0, 10):
        inflation_paths, variance_paths = Heston_Model.simulate_paths_euler(NoOfSteps, NoOfPaths) 
        P = np.append(P, np.exp(-r * T) * np.maximum(inflation_paths[:,-1] - K, 0))
        
    et = time.time()
    MC_mean = np.mean(P)
    error = np.sqrt(np.var(P) / (NoOfPaths*10))
    bias = OptionValuewithCos - MC_mean
    print('For dt = ', dt, 'the monte carlo price with exact trapezoidal and scipy package for the variance is', MC_mean, 'and the standard error is', error,' and bias', bias,'. This took', et - st,'seconds')
    


for i in range(0, 7):
    dt = T / 4**i
    NoOfSteps = 4**i
    st = time.time()
    P = np.array([])
    for i in range(0, 10):
        inflation_paths, variance_paths = Heston_Model.simulate_paths_euler(NoOfSteps, NoOfPaths) 
        P = np.append(P, np.exp(-r * T) * np.maximum(inflation_paths[:,-1] - K, 0))
        
    et = time.time()
    MC_mean = np.mean(P)
    error = np.sqrt(np.var(P) / (NoOfPaths*10))
    bias = OptionValuewithCos - MC_mean
    print('For dt = ', dt, 'the monte carlo price with exact trapezoidal and QE for the variance is', MC_mean, 'and the standard error is', error,' and bias', bias,'. This took', et - st,'seconds')

for i in range(0, 7):
    dt = T / 4**i
    NoOfSteps = 4**i
    st = time.time()
    P = np.array([])
    for i in range(0, 10):
        inflation_paths, variance_paths = Heston_Model.simulate_paths_euler(NoOfSteps, NoOfPaths) 
        P = np.append(P, np.exp(-r * T) * np.maximum(inflation_paths[:,-1] - K, 0))
        
    et = time.time()
    MC_mean = np.mean(P)
    error = np.sqrt(np.var(P) / (NoOfPaths*10))
    bias = OptionValuewithCos - MC_mean
    print('For dt = ', dt, 'the monte carlo price with exact trapezoidal and NCI for the variance is', MC_mean, 'and the standard error is', error,' and bias', bias,'. This took', et - st,'seconds')
    


"""  Multi-level Monte Carlo simulation for different mean squared errors """

errors = [0.1, 0.05, 0.02, 0.01, 0.005]

Nls = []
MLMCprices = []
Costs_MLMC = []
Costs_MC = []
M = 4

for error in errors:

    MLMC_price, m_l, v_l, Nl, L, standardMC_mean, standardMC_var = Heston_Model.MLMC(K, error)
    levels = np.linspace(0, L, L+1)

    print('the MLMC price is', MLMC_price)
    MLMCprices.append(MLMC_price)

    Nls.append(Nl)
    

    Heston_Model.plot_means(4, m_l, standardMC_mean, L)
    Heston_Model.plot_variances(4, v_l, standardMC_var, L)
    
    C1 = Nl[0] * error**2
    C2 = 2 * standardMC_var[0] 
    for i in range(1, L+1):
        C1 += Nl[i] * (M**i + M**(i-1)) * error**2
        C2 += 2 * standardMC_var[i] * M**(i) 
    Costs_MLMC.append(C1)
    Costs_MC.append(C2)
  
    


plt.figure()
plt.plot(errors, Costs_MC, marker = '*', label = r'$C_{MC}$', linestyle = '--')
plt.plot(errors, Costs_MLMC, marker = '*', label = r'$C_{MLMC}$',linestyle = '--')
plt.ylabel(r'$\epsilon^2 C$')
plt.yscale('log')
plt.xscale('log')
plt.title('Computation costs MLMC vs MC')
plt.yticks([10**i for i in range(2,9)], [f'$10^{i} $' for i in range(2,9)])
plt.xticks([10**i for i in range(-3,0)], [f'$10^{i} $' for i in range(-3,0)])
plt.xlabel(r'$\epsilon$')
plt.legend()
plt.show()


plt.figure()
for i, error in enumerate(errors):
    x = np.linspace(0, len(Nls), len(Nls))
    plt.plot(Nls[i], label = f'$\epsilon$={error}', marker = '*',linestyle = '--')
plt.yscale('log')
plt.yticks([10**i for i in range(1,10)], [f'$10^{i} $' for i in range(1,10)])
plt.title('Number of samples per level')
plt.xlabel(r'level ($\ell$)')
plt.ylabel(r'$N_{\ell}$')
plt.legend()
plt.show()

print('The MLMC prices for errors', errors, 'are', MLMCprices)


