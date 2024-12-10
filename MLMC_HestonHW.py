 # -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:22:46 2024

@author: hvermeer
"""

import numpy as np
import matplotlib.pyplot as plt
my_generator = np.random.default_rng()

class HestonHW_Model:
    def __init__(self, initial_index, initial_variance, initial_nominal, a_n, volatility_nominal, initial_real, 
                 a_r, volatility_real, mean_reversion_variance, corr_matrix, mean_variance, volatility_var,  T):
        """  Specificies the (initial) parameters of the Heston Hull-White model. These are independent of the used simulation method.  
        
        Parameters
        -----------
        initial_index : Initial value of the inflation index.
        initial_variance : Initial value of the variance process.
        initial_nominal : Initial value of the nominal interest rate.
        a_n : Mean reversion parameter of the nominal interest rate.
        volatility_nominal : Constant volatility of the nominal interest rate.
        a_r : Mean reversion parameter of the real interest rate.
        volatility_real : Constant volatility of the real interest rate.
        initial_real : Initial value of the real interest rate.
        mean_reversion_variance : Mean reversion coefficient of the variance process (alpha).
        correlation : Correlation coefficient between the brownian motions of the inflation index and variance process. 
        mean_variance : Average level of the variance process.
        volatility_of_variance : Volatility parameter of the variance process.
        
        T : Time to maturity.
        
        """    
        self.I0 = initial_index
        self.V0 = initial_variance
        self.r_n = initial_nominal
        self.a_n = a_n
        self.sigma_n = volatility_nominal
        self.r_r = initial_real
        self.a_r = a_r
        self.sigma_r = volatility_real
        self.alpha = mean_reversion_variance
        self.corr = corr_matrix
        self.Vbar = mean_variance
        self.sigmaV = volatility_of_variance
        self.T = T     
        
    def correlated_BM(self, NoOfPaths, NoOfSteps):
        """  Constructs a 4-dimensional array of correlated standard normal variables using Cholesky decomposition. One standard normal variable per time step per sample path. 
        
        Parameters
        -----------
        NoOfPaths: Number of samples 
        NoOfSteps: Number of time steps
        
        Returns: Array of correlated standard normal variables.
        """

        C = self.corr
        
        L = np.linalg.cholesky(C)
        
        W = np.random.normal(0, 1, (4, NoOfPaths, NoOfSteps))
            
        CW = np.einsum('ij,jkl->ikl', L, W)

        return CW
    
    def B_n(self, t):
        
        B_n = (1 - np.exp(-self.a_n * (self.T - t))) / self.a_n
        
        return B_n
    
    def B_r(self, t):
        
        B_r = (1 - np.exp(-self.a_r * (self.T - t))) / self.a_r
        
        return B_r
    
    def phi(self, t):
        
        phi = np.sqrt(self.Vbar * (1 - np.exp(-self.alpha * t)) + self.V0 * np.exp(- self.alpha * t) - self.sigmaV * (1 - np.exp(-self.alpha * t)) * (self.Vbar * np.exp(self.alpha * t) - self.Vbar + 2 * self.V0) / (8 * self.alpha * (self.Vbar * np.exp(self.alpha * t - self.Vbar + self.V0))))
        
        return phi
    
    def eta(self, t):
        
        rho_In = self.corr[0][2]
        rho_Ir = self.corr[0][3]
        rho_nr = self.corr[1][2]
        
        
        eta = (rho_Ir * self.sigma_r * self.B_r(t) - rho_In * self.sigma_n * self.B_n(t)) * self.phi(t) + rho_nr * self.sigma_n * self.sigma_r * self.B_n(t) * self.B_r(t) - (self.sigma_n**2 * self.B_n(t)**2 + self.sigma_r**2 * self.B_r(t)**2) / 2
        
        return eta
        
    def C(self, u, t):
        xi = self.alpha - self.sigmaV * self.corr[0][1] * u * 1j
        
        d = np.sqrt(xi**2 + self.sigmaV**2 * (u**2 + 1j * u))
        
        g = (xi - d) / (xi + d)
        
        C = (xi - d) * (1 - np.exp(-d * t)) / (self.sigmaV**2 * (1 - g * np.exp(-d * t)))
        
        return C
    
    def cf_HHW(self, u, t):
        """  The characteristic function of the Heston Hull-White (1993) model 
        
        """
        
        tau = self.T - t
        
        """  Function B  """
        
        B = u * 1j
        
        """  Complex-valued C function (of Heston type) """
        
        C = self.C(u, tau)
        
        """  Function A using simpsons integration """

        first_term = (self.alpha * self.Vbar - self.corr[1][2] * self.sigmaV * self.sigma_n * self.phi(0) * (self.B_n(0)) * (1 - u * 1j) - self.corr[1][3] * self.sigmaV * self.phi(0) * (self.B_r(0)) * u * 1j) * self.C(u, 0) + (u**2 + u * 1j) * self.eta(0)
        
        second_term = (self.alpha * self.Vbar - self.corr[1][2] * self.sigmaV * self.sigma_n * self.phi(tau/2) * (self.B_n(tau / 2)) * (1 - u * 1j) - self.corr[1][3] * self.sigmaV * self.phi(tau / 2) * (self.B_r(tau / 2))* u * 1j) * self.C(u, tau / 2) + (u**2 + u * 1j) * self.eta(tau / 2)
        
        third_term = (self.alpha * self.Vbar - self.corr[1][2] * self.sigmaV * self.sigma_n * self.phi(tau) * (self.B_n(tau)) * (1 - u * 1j) - self.corr[1][3] * self.sigmaV * self.phi(tau) * (self.B_r(tau))* u * 1j) * self.C(u, tau) + (u**2 + u * 1j) * self.eta(tau)
        
        A = tau * (first_term + 4 * second_term + third_term) / 6
        
        """  Combine into the characteristic function  """
        
        cf = np.exp(A + B * np.log(self.I0) + C * self.V0)
        
        return cf
      
    def OptionValueCos(self, K, N, a, b, T):
        """  Calculates the analytical value of a call option on the Heston model using its characteristic function  
        
        Parameters
        -----------
        K : Strike price
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
        cf = self.cf_HHW(u, 0)

        
        # Iterate through the strike prices.
        for m in range(0, np.size(K)):
            x = np.log(self.I0/K[m])
            Term = np.exp(1j * k * np.pi * (x-a) / bma)
            Fk = np.real(cf * Term)
            Fk[0] = 0.5 * Fk[0]

            V_CALL[m] = K[m] * np.sum(Fk * Uk(k, a, b, 'call'))
            V_PUT[m] = K[m] * np.sum(Fk * Uk(k, a, b, 'put')) 

            V_PCP[m] = V_PUT[m] + self.I0 * np.exp(-q * T) - K[m] * np.exp(-(self.r_n - self.r_r) * T)
            
        return V_PCP, cf
     
    def euler_step_truncated(self, index, v, r_n, r_r, z1, z2, z3, z4, dt, Wdt, theta_n, theta_r):
        """   Performs one iteration of the full truncated euler discretization step for the log heston model 
        
        Parameters
        -----------
        index : current value of the inflation index
        v : current value of the variance process
        r_n : current value of the nominal interest rate
        r_r : current value of the real interest rate
        z1 : correlated sample from standard normal distribution for the inflation index
        z2 : correlated sample from standard normal distribution for the variance process
        z3 : correlated sample from standard normal distribution for the nominal interest rate
        z4 : correlated sample from standard normal distribution for the real interest rate
        dt : time step 
        Wdt : time step used for the brownian motion
        theta_n : values of theta for the nominal interest rate
        thata_r : values of theta for the real interest rate
        
        """
        r = r_n - r_r
        v_sq = np.sqrt(np.maximum(v, 0))
        Wdt_sq = np.sqrt(Wdt)
        
        index = index * np.exp((r - 0.5 * v) * dt + v_sq * Wdt_sq * z1)
        
        v = v + self.alpha * (self.Vbar - np.maximum(v, 0)) * dt + self.sigmaV * v_sq * Wdt_sq * z2
        v = np.maximum(v, 0)
        
        r_n = r_n + (theta_n - self.a_n *r_n) * dt + self.sigma_n * Wdt_sq * z3
        
        r_r = r_r + (theta_r - self.corr[0][3] * self.sigma_r * v_sq - self.a_n * r_r) * dt + self.sigma_r * Wdt_sq * z4
        
        return index, v, r_n, r_r
    
    def exact_trapezoidal_step_fine(self, index, v, r_n, r_r, z1, z3, z4, dt, alpha_n_t, alpha_n_s, alpha_r_t, alpha_r_s):
        """   Performs one iteration using exact simulation techniques for the fine level in the MLMC algorithm
        
        Parameters
        -----------
        index : current value of the inflation index
        v : current value of the variance process
        r_n : current value of the nominal interest rate
        r_r : current value of the real interest rate
        z1 : correlated sample from standard normal distribution for the inflation index
        z2 : correlated sample from standard normal distribution for the variance process
        z3 : correlated sample from standard normal distribution for the nominal interest rate
        z4 : correlated sample from standard normal distribution for the real interest rate
        dt : time step 
        alpha_n_t : value of alpha at time t + dt for the nominal interest rate
        alpha_n_s : value of alpha at time t for the nominal interest rate
        alpha_r_t : value of alpha at time t + dt for the real interest rate
        alpha_n_s : value of alpha at time t for the real interest rate
        
        """
        
        
        d = 4 * self.Vbar * self.alpha / self.sigmaV ** 2
        lam = v * 4 * self.alpha * np.exp(-self.alpha * dt) / (self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)))
        
        ncxd = my_generator.noncentral_chisquare(d, lam)
        
        v_t = ncxd * self.sigmaV**2 * (1 - np.exp(-self.alpha * dt)) / (4 * self.alpha) 
        
        v_trap = (v_t + v) * dt / 2
        
        v_sq_trap = (np.sqrt(v_t) + np.exp(-self.a_r * dt) * np.sqrt(v)) * dt / 2
        
        r_nominal_mean = r_n * np.exp(-self.a_n * dt) + alpha_n_t - alpha_n_s * np.exp(-self.a_n * dt) 
        r_real_mean = r_n * np.exp(-self.a_r * dt) + alpha_r_t - alpha_r_s * np.exp(-self.a_r * dt) - self.corr[0][3] * self.sigma_r * v_sq_trap * dt
        r_nominal_sd = np.sqrt(self.sigma_n**2 / (2 * self.a_n) * (1 - np.exp(-2 * self.a_n * dt)))
        r_real_sd = np.sqrt(self.sigma_r**2 / (2 * self.a_r) * (1 - np.exp(-2 * self.a_r * dt)))
        
        r_nominal = r_nominal_mean + r_nominal_sd * z3
        r_real = r_real_mean + r_real_sd * z4
        
        r_nominal_trap = (r_n + r_nominal) * dt / 2
        r_real_trap = (r_r + r_real) * dt / 2
    
        term1 = (r_nominal_trap - r_real_trap) - (self.corr[0][1] * self.alpha * self.Vbar / self.sigmaV) * dt
        term2 = (self.corr[0][1] * self.alpha /self.sigmaV - 1/2) * v_trap
        term3 = self.corr[0][1] * (v_t - v) / self.sigmaV
        term4 = np.sqrt(1 - self.corr[0][1] ** 2) * np.sqrt(v_trap) * z1
        
        index = index * np.exp(term1 + term2 + term3 + term4)
        
        return index, v_t, r_nominal, r_real
    
    def exact_trapezoidal_step_coarse(self, index, v1, v2, r_n, r_r, z1, z3, z4, rdt, dt, alpha_n_t, alpha_n_s, alpha_r_t, alpha_r_s):
        """   Performs one iteration using exact simulation techniques for the fine level in the MLMC algorithm
        
        Parameters
        -----------
        index : current value of the inflation index
        v1 : current value of the variance process
        v2 : sampled value of the variance process at the next step
        r_n : current value of the nominal interest rate
        r_r : current value of the real interest rate
        z1 : correlated sample from standard normal distribution for the inflation index
        z2 : correlated sample from standard normal distribution for the variance process
        z3 : correlated sample from standard normal distribution for the nominal interest rate
        z4 : correlated sample from standard normal distribution for the real interest rate
        dt : time step 
        alpha_n_t : value of alpha at time t + dt for the nominal interest rate
        alpha_n_s : value of alpha at time t for the nominal interest rate
        alpha_r_t : value of alpha at time t + dt for the real interest rate
        alpha_n_s : value of alpha at time t for the real interest rate
        
        """
        v_trap = (v1 + v2) * dt / 2
        v_sq_trap = (np.sqrt(v1) + np.exp(-self.a_r * dt) * np.sqrt(v2)) * dt / 2
        
        
        r_nominal_mean = r_n * np.exp(-self.a_n * dt) + alpha_n_t - alpha_n_s * np.exp(-self.a_n * dt) 
        r_real_mean = r_n * np.exp(-self.a_r * dt) + alpha_r_t - alpha_r_s * np.exp(-self.a_r * dt) - self.corr[0][3] * self.sigma_r * v_sq_trap * dt
        r_nominal_sd = np.sqrt(self.sigma_n**2 / (2 * self.a_n) * (1 - np.exp(-2 * self.a_n * dt)))
        r_real_sd = np.sqrt(self.sigma_r**2 / (2 * self.a_r) * (1 - np.exp(-2 * self.a_r * dt)))
        
        r_nominal = r_nominal_mean + r_nominal_sd * z3
        r_real = r_real_mean + r_real_sd * z4
        
        r_nominal_trap = (r_n + r_nominal) * rdt / 2
        r_real_trap = (r_r + r_real) * rdt / 2
    
        term1 = (r_nominal_trap - r_real_trap) - (self.corr[0][1] * self.alpha * self.Vbar / self.sigmaV) * dt
        term2 = (self.corr[0][1] * self.alpha /self.sigmaV - 1/2) * v_trap
        term3 = self.corr[0][1] * (v2 - v1) / self.sigmaV
        term4 = np.sqrt(1 - self.corr[0][1] ** 2) * np.sqrt(v_trap) * z1
        
        index = index * np.exp(term1 + term2 + term3 + term4)
        
        return index, r_nominal, r_real
    
    
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
        time_values = np.linspace(0, T, NoOfSteps + 1)
        
        f_n = 0.03
        f_r = 0.03
        theta_n = a_n * f_n + (self.sigma_n**2 / (2*self.a_n)) * (1 - np.exp(-self.a_n*(time_values))) # term structure
        theta_r = a_r * f_r + (self.sigma_r**2 / (2*self.a_r)) * (1 - np.exp(-self.a_r*(time_values)))# term structure
        
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        r_nominal = np.zeros((NoOfPaths, NoOfSteps + 1))
        r_real = np.zeros((NoOfPaths, NoOfSteps + 1))
        
        
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        r_nominal[:, 0] = self.r_n
        r_real[:, 0] = self.r_r
        
        if l == 0:
            inflation_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            variance_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            r_nominal_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            r_real_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            
        else:
            inflation_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            variance_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            r_nominal_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            r_real_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            
        inflation_prev[:, 0] = self.I0
        variance_prev[:, 0] = self.V0
        r_nominal_prev[:, 0] = self.r_n
        r_real_prev[:, 0] = self.r_r
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        W_V = correlated_normal[1, :, :]
        W_r_n = correlated_normal[2, :, :]
        W_r_r = correlated_normal[3, :, :]
        
        if l == 0:
            for i in range(1, NoOfSteps + 1):
                new_inf, new_var, new_r_n, new_r_r = self.euler_step_truncated(inflation[:, i-1], variance[:, i-1], r_nominal[:, i-1], r_real[:, i-1], W_I[:, i-1], W_V[:, i-1], W_r_n[:,i-1], W_r_r[:, i-1], dt, dt, theta_n[i], theta_r[i])
                inflation[:, i] = new_inf
                variance[:, i] = new_var
                r_nominal[:, i] = new_r_n
                r_real[:, i] = new_r_r
                
            return inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev
            
        else:
            for i in range(1, int(NoOfSteps/M) + 1):
                W_I_prev = np.zeros((NoOfPaths))
                W_V_prev = np.zeros((NoOfPaths))
                W_r_n_prev = np.zeros((NoOfPaths))
                W_r_r_prev = np.zeros((NoOfPaths))
                
                for m in range(0, M):
                    new_inf, new_var, new_r_n, new_r_r = self.euler_step_truncated(inflation[:, (i-1)*M + m], variance[:, (i-1)*M + m], r_nominal[:, (i-1)*M + m], r_real[:, (i-1)*M + m], W_I[:, (i-1)*M + m], W_V[:, (i-1)*M + m], W_r_n[:, (i-1)*M + m], W_r_r[:, (i-1)*M + m], dt, dt, theta_n[(i-1)*M + m + 1], theta_r[(i-1)*M + m + 1])
                    inflation[:, (i-1)*M + m + 1] = new_inf
                    variance[:, (i-1)*M + m + 1] = new_var
                    r_nominal[:, (i-1)*M + m + 1] = new_r_n
                    r_real[:, (i-1)*M + m + 1] = new_r_r

                    W_I_prev += W_I[:, (i-1)*M + m]
                    W_V_prev += W_V[:, (i-1)*M + m]
                    W_r_n_prev += W_I[:, (i-1)*M + m]
                    W_r_r_prev += W_V[:, (i-1)*M + m]

                
                # W_I_prev = W_I_prev / np.sqrt(M)
                # W_V_prev = W_V_prev / np.sqrt(M)


                new_inf_prev, new_var_prev, new_r_nominal_prev, new_r_real_prev = self.euler_step_truncated(inflation_prev[:, i-1], variance_prev[:, i-1], r_nominal[:, i-1], r_real[:, i-1], W_I_prev, W_V_prev, W_r_n_prev, W_r_r_prev, M * dt, dt, theta_n[i], theta_r[i])
                
                inflation_prev[:, i] = new_inf_prev
                variance_prev[:, i] = new_var_prev
                r_nominal_prev[:, i] = new_r_nominal_prev
                r_real_prev[:, i] = new_r_real_prev
            
            return inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev
    
    def simulate_exact_trapezoidal_MLMC(self, NoOfPaths, M, l, h_0):
        """ Simulate the complete paths of the inflation and variance process at the coarse and fine levels using the exact simulation
        scheme. Used specifically for multilevel Monte Carlo.
        
        Parameters
        -----------
        NoOfPaths: number of samples.
        M : difference in size of the time step between fine level and coarse level.
        l : current level.
        h_0 : adjust the size of the timestep in the first level.
        
        """
        
        
        NoOfSteps = h_0 * M**l 
        
        dt = self.T / NoOfSteps
        time_values = np.linspace(0, T, NoOfSteps + 1)
        
        f_n = 0.03
        f_r = 0.03
        theta_n = a_n * f_n + (self.sigma_n**2 / (2*self.a_n)) * (1 - np.exp(-self.a_n*(time_values))) # term structure
        theta_r = a_r * f_r + (self.sigma_r**2 / (2*self.a_r)) * (1 - np.exp(-self.a_r*(time_values)))# term structure
        alpha_n = f_n + (self.sigma_n**2 / (2 * (self.a_n)**2)) * (1 - np.exp(-self.a_n * (time_values)))**2
        alpha_r = f_r + (self.sigma_n**2 / (2 * (self.a_r)**2)) * (1 - np.exp(-self.a_r * (time_values)))**2
        
        inflation = np.zeros((NoOfPaths, NoOfSteps + 1))
        variance = np.zeros((NoOfPaths, NoOfSteps + 1))
        r_nominal = np.zeros((NoOfPaths, NoOfSteps + 1))
        r_real = np.zeros((NoOfPaths, NoOfSteps + 1))
        
        
        inflation[:, 0] = self.I0
        variance[:, 0] = self.V0
        r_nominal[:, 0] = self.r_n
        r_real[:, 0] = self.r_r
        
        if l == 0:
            inflation_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            variance_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            r_nominal_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            r_real_prev = np.zeros((NoOfPaths, NoOfSteps + 1))
            
        else:
            inflation_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            variance_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            r_nominal_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            r_real_prev = np.zeros((NoOfPaths, int(NoOfSteps/M) + 1))
            
        inflation_prev[:, 0] = self.I0
        variance_prev[:, 0] = self.V0
        r_nominal_prev[:, 0] = self.r_n
        r_real_prev[:, 0] = self.r_r
        
        correlated_normal = self.correlated_BM(NoOfPaths, NoOfSteps)
        
        W_I = correlated_normal[0, :, :]
        W_V = correlated_normal[1, :, :]
        W_r_n = correlated_normal[2, :, :]
        W_r_r = correlated_normal[3, :, :]

        if l == 0:
            for i in range(1, NoOfSteps + 1):
                new_inf, new_var, new_r_n, new_r_r = self.exact_trapezoidal_step_fine(inflation[:, i-1], variance[:, i-1], r_nominal[:, i-1], r_real[:, i-1], W_I[:, i-1], W_r_n[:, i-1], W_r_r[:, i-1], dt, alpha_n[i], alpha_n[i-1], alpha_r[i], alpha_r[i-1])
                inflation[:, i] = new_inf
                variance[:, i] = new_var
                r_nominal[:, i] = new_r_n
                r_real[:, i] = new_r_r
                
            return inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev
            
        else:
            for i in range(1, int(NoOfSteps/M) + 1):
                W_I_prev = np.zeros((NoOfPaths))
                W_V_prev = np.zeros((NoOfPaths))
                W_r_n_prev = np.zeros((NoOfPaths))
                W_r_r_prev = np.zeros((NoOfPaths))
                
                for m in range(0, M):
                    new_inf, new_var, new_r_n, new_r_r = self.exact_trapezoidal_step_fine(inflation[:, (i-1)*M + m], variance[:, (i-1)*M + m], r_nominal[:, (i-1)*M + m], r_real[:, (i-1)*M + m], W_I[:, (i-1)*M + m], W_r_n[:, (i-1)*M + m], W_r_r[:, (i-1)*M + m], dt, alpha_n[(i-1)*M + m], alpha_n[(i-1)*M + m-1], alpha_r[(i-1)*M + m], alpha_r[(i-1)*M + m-1])
                    inflation[:, (i-1)*M + m + 1] = new_inf
                    variance[:, (i-1)*M + m + 1] = new_var
                    r_nominal[:, (i-1)*M + m + 1] = new_r_n
                    r_real[:, (i-1)*M + m + 1] = new_r_r

                    W_I_prev += W_I[:, (i-1)*M + m]
                    W_V_prev += W_V[:, (i-1)*M + m]
                    W_r_n_prev += W_I[:, (i-1)*M + m]
                    W_r_r_prev += W_V[:, (i-1)*M + m]

                
                W_I_prev = W_I_prev / np.sqrt(M)
                W_V_prev = W_V_prev / np.sqrt(M)
                W_r_n_prev =  W_r_n_prev / np.sqrt(M)
                W_r_r_prev = W_r_r_prev / np.sqrt(M)


                new_inf_prev, new_r_nominal_prev, new_r_real_prev = self.exact_trapezoidal_step_coarse(inflation[:, i-1], variance[:, (i-1) * M], variance[:, i * M], r_nominal[:, i-1], r_real[:, i-1], W_I_prev, W_r_n_prev, W_r_r_prev, dt, M * dt, alpha_n[i], alpha_n[i-1], alpha_r[i], alpha_r[i-1])
                
                inflation_prev[:, i] = new_inf_prev
                r_nominal_prev[:, i] = new_r_nominal_prev
                r_real_prev[:, i] = new_r_real_prev
            
            return inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev
    
    def prices_MLMC(self, K, inflation, r_nominal, r_real, inflation_prev, r_nominal_prev, r_real_prev, steps_fine, steps_coarse):
        """ Calculates the value of the payoff function at the coarse level and the fine level.
        
        Parameters
        -----------
        K : strike price
        inflation : inflation index paths at the fine level
        inflation_prev : inflation index paths at the coarse level
        
        """
        
        discount_factor_fine = self.T * r_nominal[:, 0] / (2 * steps_fine) + self.T * r_nominal[:, -1] / (2 * steps_fine)
        discount_factor_coarse = self.T * r_nominal_prev[:, 0] / (2 * steps_fine) + self.T * r_nominal_prev[:, -1] / (2 * steps_fine)

        for i in range(1, len(r_nominal[0, :]) - 1):
            discount_factor_fine += r_nominal[:, i] * self.T / steps_fine
            
        for i in range(1, len(r_nominal_prev[0, :]) - 1):
            discount_factor_coarse += r_nominal_prev[:, i] * self.T / steps_coarse

        P_f = np.exp(-discount_factor_fine) * np.maximum(inflation[:, -1] - K, 0)
        P_c = np.exp(-discount_factor_coarse) * np.maximum(inflation_prev[:, -1] - K, 0)

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
                    if dNl[l]*M**l > 10**7:
                        dNl[l] = 10**7/(M**l)
                    inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev = self.simulate_paths_euler_MLMC(int(dNl[l]), M, l, h_0)
                    # inflation, variance, r_nominal, r_real, inflation_prev, variance_prev, r_nominal_prev, r_real_prev = self.simulate_paths_euler_MLMC(int(dNl[l]), M, l, h_0)

                    P_f, P_c = self.prices_MLMC(K, inflation, r_nominal, r_real, inflation_prev, r_nominal_prev, r_real_prev, M**l, M**(l-1))

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
            print(m_l)
            v_l = np.maximum(0, sum2 / Nl -  m_l**2)
            print(v_l)
            standardMC_mean = np.abs(sum3 / Nl)
            standardMC_var = sum4 / Nl -  standardMC_mean**2
            
            # Estimate optimal samples.
            
            h_l = np.array([T / (M**l) for l in range(0, L+1)])
 
            N_s = np.ceil(2 * np.sqrt(v_l * h_l) * np.sum(np.sqrt(v_l / h_l)) / xi**2)
            
            dNl = np.maximum(0, N_s - Nl)
            print(dNl)
            
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
    
    def plot_means(self, base, m_l, m_MC, L):
        
        levels = np.linspace(0, L, L+1)
        
        log_means_MLMC = np.emath.logn(4, m_l)
        log_means_MC = np.emath.logn(4, m_MC)
        
        plt.figure()
        plt.plot(levels, log_means_MC, marker = '*', label = r'$E(\hat{P}_{\ell})$',linestyle = '--')
        plt.plot(levels, log_means_MLMC, marker = '*', label = r'$E(\hat{P}_{\ell} -\hat{P}_{\ell-1})$',linestyle = '--')
        plt.title('Expectation MLMC vs MC')
        plt.ylabel(r'$\log_{4}(|mean|)$')
        plt.xlabel(r'level ($\ell$)')
        plt.legend()
        plt.show()
       
    def plot_variances(self, base, v_l, Var_MC, L):
        levels = np.linspace(0, L, L+1)
        
        log_variances_MLMC = np.emath.logn(4, v_l)
        log_variances_MC = np.emath.logn(4, Var_MC)
        
        plt.figure()
        plt.plot(levels, log_variances_MC, marker = '*', label = r'$Var{V}(\hat{P}_{\ell})$', linestyle = '--')
        plt.plot(levels, log_variances_MLMC, marker = '*', label = r'$Var(\hat{P}_{\ell} -\hat{P}_{\ell-1})$',linestyle = '--')
        plt.title('Variance MLMC vs MC')
        plt.ylabel(r'$\log_{4}(variance)$')
        plt.xlabel(r'level ($\ell$)')
        plt.legend()
        plt.show()
    
    def plot_paths(self, inflation_paths, variance_paths, nominal_interest_rate_paths, real_interest_rate_paths, time_values):
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
        
        for i in range(nominal_interest_rate_paths.shape[0]):
            plt.plot(time_values, nominal_interest_rate_paths[i], label=f'Variance Path {i+1}')

        plt.title('Simulated Nominal Interest Rate Paths')
        plt.xlabel('Time')
        plt.ylabel('Nominal Interest Rate')
        plt.show()
        
        for i in range(real_interest_rate_paths.shape[0]):
            plt.plot(time_values, real_interest_rate_paths[i], label=f'Variance Path {i+1}')

        plt.title('Simulated Real Interest Rate Paths')
        plt.xlabel('Time')
        plt.ylabel('Real Interest Rate')
        plt.show()
        
    def truncationRange(self, L, T):
        mu = self.r_n - self.r_r
        sigma = self.V0
        lm = self.alpha
        v_bar = self.Vbar
        rho = self.corr[0][1]
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


initial_index = 100

initial_nominal = 0.05
initial_real = 0.04
a_n = 0.03 # mean reversion nominal
a_r = 0.03 # mean revresion real


nominal_volatility = 0.0089
real_volatility = 0.0084

rho_IV = -0.89
rho_In = 0.36
rho_Ir = -0.29
rho_Vn = -0.323
rho_Vr = 0.261
rho_nr = 0.78
corr_matrix = np.array([[1, rho_IV, rho_In, rho_Ir], [rho_IV, 1, rho_Vn, rho_Vr], [rho_In, rho_Vn, 1, rho_nr],[rho_Ir, rho_Vr, rho_nr, 1]])

mean_variance = 0.04
volatility_of_variance = 0.06
initial_variance = 0.04
mean_reversion_variance = 0.3

T = 1.0
K = 100



print('Feller condition is satisfied if > 1 :', 2*mean_variance*mean_reversion_variance / volatility_of_variance**2)

HestonHW_Model = HestonHW_Model(initial_index, initial_variance, initial_nominal, a_n, nominal_volatility, initial_real, a_r, real_volatility, mean_reversion_variance, corr_matrix, mean_variance, volatility_of_variance, T)
 



L = 120
a,b = HestonHW_Model.truncationRange(L, T)

K = [100]
q=0
N = 1024



OptionValuewithCos, cf = HestonHW_Model.OptionValueCos(K, N, a, b, T)

print('Option value using COS method', OptionValuewithCos)






K = 100
errors = [0.1, 0.05, 0.02, 0.01, 0.005]

Nls = []
MLMCprices = []
Costs_MLMC = []
Costs_MC = []
M = 4

for error in errors:

    MLMC_price, m_l, v_l, Nl, L, standardMC_mean, standardMC_var = HestonHW_Model.MLMC(K, error)
    levels = np.linspace(0, L, L+1)

    print('the MLMC price is', MLMC_price)
    MLMCprices.append(MLMC_price)

    Nls.append(Nl)
    

    HestonHW_Model.plot_means(4, m_l, standardMC_mean, L)
    HestonHW_Model.plot_variances(4, v_l, standardMC_var, L)
    
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
plt.yticks([10**i for i in range(2,7)], [f'$10^{i} $' for i in range(2,7)])
plt.xticks([10**i for i in range(-3,0)], [f'$10^{i} $' for i in range(-3,0)])
plt.xlabel(r'$\epsilon$')
plt.legend()
plt.show()


plt.figure()
for i, error in enumerate(errors):
    x = np.linspace(0, len(Nls), len(Nls))
    plt.plot(Nls[i], label = f'$\epsilon$={error}', marker = '*',linestyle = '--')
plt.yscale('log')
plt.yticks([10**i for i in range(1,8)], [f'$10^{i} $' for i in range(1,8)])
plt.title('Number of samples per level')
plt.xlabel(r'level ($\ell$)')
plt.ylabel(r'$N_{\ell}$')
plt.legend()
plt.show()

print('The MLMC prices for errors', errors, 'are', MLMCprices)

# # Parameters for Heston model
# initial_inflation = 100  # Initial inflation index level
# mean_reversion_inflation = 2.0
# long_term_volatility = 0.04
# volatility_inflation = 0.3
# correlation_inflation = -0.5
# initial_volatility_inflation = 0.04
# risk_free_rate = 0.05
# time_to_maturity = 1.0
# n_steps = 100
# n_paths = 1000
# delta_t = time_to_maturity / n_steps

# # Create Heston model instance for inflation dynamics
# heston_model = HestonModel(initial_inflation, mean_reversion_inflation, volatility_inflation, correlation_inflation, long_term_volatility, initial_volatility_inflation, risk_free_rate, delta_t)

# # Parameters for Hull-White model
# mean_reversion_hw = 0.1
# volatility_hw = 0.02
# short_rate = 0.03

# # Create Hull-White model instance for nominal interest rate
# hw_model_nominal = HullWhiteModel(mean_reversion_hw, volatility_hw, short_rate, delta_t)

# # Create Hull-White model instance for real interest rate
# hw_model_real = HullWhiteModel(mean_reversion_hw, volatility_hw, short_rate, delta_t)

# # Simulate paths for the inflation index
# inflation_paths, volatility_paths = heston_model.simulate_path(n_steps, n_paths)

# # Simulate paths for nominal and real interest rates
# nominal_interest_rate_paths = hw_model_nominal.simulate_path(n_steps, n_paths)
# real_interest_rate_paths = hw_model_real.simulate_path(n_steps, n_paths)

# # Output the simulated paths
# print("Simulated inflation index values:")
# print(inflation_paths)
# print("Simulated nominal interest rate paths:")
# print(nominal_interest_rate_paths)
# print("Simulated real interest rate paths:")
# print(real_interest_rate_paths)
