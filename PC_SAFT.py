import numpy as np
import numdifftools as nd
from scipy.optimize import root


class PCSAFT:

    a_ni = np.array([[0.9105631445, -0.3084016918, -0.0906148351],
                     [0.6361281449, 0.1860531159, 0.4527842806],
                     [2.6861347891, -2.5030047259, 0.5962700728],
                     [-26.547362491, 21.419793629, -1.7241829131],
                     [97.759208784, -65.255885330, -4.1302112531],
                     [-159.59154087, 83.318680481, 13.776631870],
                     [91.297774084, -33.746922930, -8.6728470368]])
    a_ni = a_ni.T

    b_ni = np.array([[0.7240946941, -0.5755498075, 0.0976883116],
                     [2.2382791861, 0.6995095521, -0.2557574982],
                     [-4.0025849485, 3.8925673390, -9.1558561530],
                     [-21.003576815, -17.215471648, 20.642075974],
                     [26.855641363, 192.67226447, -38.804430052],
                     [206.55133841, -161.82646165, 93.626774077],
                     [-355.60235612, -165.20769346, -29.666905585]])
    b_ni = b_ni.T

    kb = 1.380649e-23  # J/K
    N_A = 6.0221e23  # 1/mol
    R = 8.314  # J/mol-K
    π = np.pi

    def __init__(self, T, z, m, σ, ϵ_k, k_ij, η=None, phase='liquid', P_sys=None, κ_AB=None, ϵ_AB_k=None, find_η_bool=False):

        # Parameters
        self.T = T
        self.z = z
        self.m = m
        self.k = len(σ)
        k = self.k

        if κ_AB is None:
            κ_AB = np.zeros(k)

        if ϵ_AB_k is None:
            ϵ_AB_k = np.zeros(k)

        # --------------------------------------- Intermediates --------------------------------------- #

        self.σ_ij = np.array([[1 / 2 * (σ[i] + σ[j]) for j in range(k)] for i in range(k)])
        self.ϵ_ij = np.array([[(ϵ_k[i] * ϵ_k[j]) ** (1 / 2) * (1 - k_ij[i][j]) for j in range(k)] for i in range(k)])
        self.κ_AB_ij = np.array(
            [[(κ_AB[i] * κ_AB[j]) ** (1 / 2) * ((σ[i] * σ[j]) / (1 / 2 * (σ[i] * σ[j]))) ** 3 for j in range(k)] for i
             in range(k)])
        self.ϵ_AB_ij = np.array([[(ϵ_AB_k[i] + ϵ_AB_k[j]) / 2 for j in range(k)] for i in range(k)])

        if P_sys is not None:
            self.P_sys = P_sys
            self.phase = phase
            self.find_η()
        elif P_sys is None and η is not None:
            self.η = η
        elif P_sys is None and η is None:
            if phase == 'liquid':
                self.η = .4
            elif phase == 'vapor':
                self.η = .01
            print('Warning, a default η had to be defined based on the given phase since so system pressure was given to iteratively find η')

    def m_bar(self):

        z = self.z
        m = self.m

        return sum(z * m)

    def d(self):
        T = self.T

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))

    def ρ(self):

        z = self.z
        η = self.η
        d = self.d()
        m = self.m
        return 6 / self.π * η * (sum([z[i] * m[i] * d[i] ** 3 for i in range(self.k)])) ** (-1)

    def v(self):

        ρ = self.ρ()
        return self.N_A * 10 ** -30 / ρ

    def ξ(self):

        z = self.z
        d = self.d()
        ρ = self.ρ()
        m = self.m
        return np.array([self.π / 6 * ρ * np.sum([z[i] * m[i] * d[i] ** n for i in range(self.k)]) for n in range(4)])

    def g_hs_ij(self):

        d = self.d()
        ξ = self.ξ()

        return np.array([[(1 / (1 - ξ[3])) +
                        ((d[i] * d[j] / (d[i] + d[j])) * 3 * ξ[2] / (1 - ξ[3]) ** 2) +
                        ((d[i] * d[j] / (d[i] + d[j])) ** 2 * 2 * ξ[2] ** 2 / (1 - ξ[3]) ** 3)
                        for j in range(self.k)]
                        for i in range(self.k)])

    def d_ij(self):

        d = self.d()
        return np.array([[1 / 2 * (d[i] + d[j]) for j in range(self.k)] for i in range(self.k)])

    def Δ_AB_ij(self):
        T = self.T
        d_ij = self.d_ij()
        g_hs_ij = self.g_hs_ij()
        return np.array([[d_ij[i][j] ** 3 * g_hs_ij[i][j] * self.κ_AB_ij[i][j] * (np.exp(self.ϵ_AB_ij[i][j] / T) - 1)
                          for j in range(self.k)] for i in range(self.k)])

    def a_hs(self):

        ξ = self.ξ()
        return 1 / ξ[0] * (3 * ξ[1] * ξ[2] / (1 - ξ[3]) + ξ[2] ** 3 / (ξ[3] * (1 - ξ[3]) ** 2) + (
                    ξ[2] ** 3 / ξ[3] ** 2 - ξ[0]) * np.log(1 - ξ[3]))

    def a_hc(self):
        z = self.z

        k = self.k
        m = self.m
        m_bar = self.m_bar()
        g_hs_ij = self.g_hs_ij()
        a_hs = self.a_hs()

        return m_bar * a_hs - sum([z[i] * (m[i] - 1) * np.log(g_hs_ij[i][i]) for i in range(k)])

    def a_disp(self):

        T = self.T
        z = self.z
        η = self.η

        a_ni = self.a_ni
        b_ni = self.b_ni
        π = self.π
        k = self.k
        ρ = self.ρ()
        m = self.m
        m̄ = self.m_bar()
        ϵ_ij = self.ϵ_ij
        σ_ij = self.σ_ij

        a = a_ni[0] + (m̄ - 1) / m̄ * a_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * a_ni[2]
        b = b_ni[0] + (m̄ - 1) / m̄ * b_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * b_ni[2]

        I1 = [a[i] * η ** i for i in range(7)]
        I2 = [b[i] * η ** i for i in range(7)]

        I1 = np.sum(I1)
        I2 = np.sum(I2)

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_1 = Σ_i

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) ** 2 * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_2 = Σ_i

        C1 = (1 + m̄ * (8 * η - 2 * η ** 2) / (1 - η) ** 4 + (1 - m̄) * (
                    20 * η - 27 * η ** 2 + 12 * η ** 3 - 2 * η ** 4) / ((1 - η) * (2 - η)) ** 2) ** -1

        return np.round(-2 * π * ρ * I1 * Σ_1 - π * ρ * m̄ * C1 * I2 * Σ_2, 15)

    def a_assoc(self):

        z = self.z
        k = self.k
        ρ = self.ρ()
        Δ_AB_ij = self.Δ_AB_ij()

        def XA_find(XA_guess, n, Δ_AB_ij, ρ, z):
            # m = int(XA_guess.shape[1] / n)
            AB_matrix = np.asarray([[0., 1.],
                                    [1., 0.]])
            Σ_2 = np.zeros((n,), dtype='float_')
            XA = np.zeros_like(XA_guess)

            for i in range(n):
                Σ_2 = 0 * Σ_2
                for j in range(n):
                    Σ_2 += z[j] * (XA_guess[j, :] @ (Δ_AB_ij[i][j] * AB_matrix))
                XA[i, :] = 1 / (1 + ρ * Σ_2)

            return XA

        a_sites = 2  # 2B association?
        i_assoc = np.nonzero(ϵ_AB_k)
        n_assoc = len(i_assoc)
        if n_assoc == 0 or n_assoc == 1:
            return 0
        XA = np.zeros((n_assoc, a_sites), dtype='float_')

        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, n_assoc, Δ_AB_ij, ρ, z[i_assoc])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten()

        # print([z[i] * sum([np.log(XA[j] - 1 / 2 * XA[j] + 1 / 2) for j in range(len(XA))]) for i in range(k)])

        return sum([z[i] * sum([np.log(XA[j] - 1 / 2 * XA[j] + 1 / 2) for j in range(len(XA))]) for i in range(k)])

    def a_ion(self):
        return 0

    def a_res(self):

        a_hc = self.a_hc()
        a_disp = self.a_disp()
        a_assoc = self.a_assoc()
        a_ion = self.a_ion()

        return a_hc + a_disp + a_assoc + a_ion

    def da_dη(self):

        η = self.η

        self.η_og = self.η

        def f(η_diff):
            self.η = η_diff
            return self.a_res()

        self.η = self.η_og

        return nd.Derivative(f)(η)

    def da_dx(self):

        z = self.z
        self.z_og = self.z

        da_dx = []
        for k in range(len(z)):

            def f(zk):

                z_new = []
                for i in range(len(z)):
                    if i == k:
                        z_new.append(zk)
                    else:
                        z_new.append(z[i])

                self.z = z_new
                return self.a_res()

            self.z = self.z_og

            da_dx.append(nd.Derivative(f)(z[k]))
        self.z = self.z_og

        return np.array(da_dx)

    def da_dT(self):

        T = self.T
        self.T_og = self.T

        def f(T_diff):
            self.T = T_diff
            return self.a_res()

        self.T = self.T_og

        return nd.Derivative(f)(T)

    def Z(self):
        η = self.η

        da_dη = self.da_dη()

        self.η = self.η_og

        return 1 + η * da_dη

    def P(self):

        T = self.T
        Z = self.Z()
        ρ = self.ρ()
        kb = self.kb

        return Z*kb*T*ρ*10**30

    def find_η(self):

        def f(ηg):
            self.η = float(ηg)
            P = self.P()
            P_sys = self.P_sys
            return (P - P_sys)/100000

        phase = self.phase
        if phase == 'liquid':
            ηg = .5
        elif phase == 'vapor':
            ηg = 10e-10
        else:
            print('Phase spelling probably wrong or phase is missing')
            ηg = .01
        η = root(f, np.array([ηg])).x[0]

        self.η = η

    def h_res(self):

        T = self.T
        self.T_og = self.T

        Z = self.Z()
        da_dT = self.da_dT()

        self.T = self.T_og

        return -T * da_dT + (Z - 1)

    def s_res(self):

        T = self.T
        self.T_og = self.T

        a_res = self.a_res()
        Z = self.Z()
        da_dT = self.da_dT()

        self.T = self.T_og

        return -T * (da_dT(T) + a_res / T) + np.log(Z)

    def g_res(self):
        a_res = self.a_res()
        Z = self.Z()
        return a_res + (Z - 1) - np.log(Z)

    def μ_res(self):

        z = self.z
        T = self.T

        a_res = self.a_res()
        Z = self.Z()
        da_dx = self.da_dx()

        Σ = np.sum([z[j] * da_dx[j] for j in range(len(z))])

        μ_res = [(a_res + (Z - 1) + da_dx[i] - Σ) * self.kb * T for i in range(len(z))]

        return np.array(μ_res)

    def φ(self):

        T = self.T
        μ_res = self.μ_res()
        Z = self.Z()

        return np.exp(μ_res / self.kb / T - np.log(Z))


def BPT(T, x, m, σ, ϵ_k, k_ij, yg, Pg, κ_AB=None, ϵ_AB_k=None):

    def f(z):
        y1, y2, y3, P = z

        y = [y1, y2, y3]

        mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        mix_l.find_η()
        mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        mix_v.find_η()

        φ_l = mix_l.φ()
        φ_v = mix_v.φ()

        eqs = [(y[i] * φ_v[i] - x[i] * φ_l[i]) for i in range(len(y))]
        eqs.append(1 - np.sum([y[i] for i in range(len(y))]))

        return eqs

    zg = yg
    zg.append(Pg)

    ans = root(f, zg).x

    y = ans[:-1]
    P = ans[-1]

    return y, P

#%%

#
# T = 233.15
# x = np.array([.1, .3, .6])  # Mole fraction
# m = np.array([1, 1.6069, 2.0020])  # Number of segments
# σ = np.array([3.7039, 3.5206, 3.6184])  # Temperature-Independent segment diameter σ_i (Aᵒ)
# ϵ_k = np.array([150.03, 191.42, 208.11])  # Depth of pair potential / Boltzmann constant (K)
# k_ij = np.array([[0.00E+00, 3.00E-04, 1.15E-02],
#                  [3.00E-04, 0.00E+00, 5.10E-03],
#                  [1.15E-02, 5.10E-03, 0.00E+00]])
# κ_AB = np.array([0, 0, 0])
# ϵ_AB_k = np.array([0, 0, 0])
# P_sys = 916372.96423737
# mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P_sys)
# yg = [.67, .23, .10]
# Pg = P_sys
# y, P = BPT(T, x, m, σ, ϵ_k, k_ij, yg, Pg)
# print(y, P)

#%%

MW_CO2 = 44.01/1000
MW_MEA = 61.08/1000
MW_H2O = 18.02/1000

def get_x(α, w_MEA):

    x_MEA = ((1 + α + (MW_MEA/MW_H2O))*(1-w_MEA)/w_MEA)**-1
    x_CO2 = x_MEA*α
    x_H2O = 1 - x_CO2 - x_MEA

    return [x_CO2, x_MEA, x_H2O]


T = 393 # K
w_MEA = .30
α = .4
x = get_x(α, w_MEA)
x = [.04, .1, .86]
y = [.1, .013, .818, .069]

# 2B Association Scheme
x = [.04, .1, .86]
m = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
ϵ_k = np.array([169.21, 277.174, 279.42]) # Depth of pair potential / Boltzmann constant (K)
k_ij = np.array([[0.0, .16, .065],
                 [.16, 0.0, -.18],
                 [.065, -.18, 0.0]])
κ_AB = np.array([0, .037470, .2039])
ϵ_AB_k = np.array([0, 2586.3, 2059.28])

P_sys = 109180
#
mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P_sys, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
print(mix_l.φ())

# Vapor
y = [.1, .013, .818, .069]
m = np.array([2.0729, 1.9599, 1.2053, 1.1335])  # Number of segments
σ = np.array([2.7852, 2.362, 3.3130, 3.1947])  # Temperature-Independent segment diameter σ_i (Aᵒ)
ϵ_k = np.array([169.21, 279.42, 90.96, 114.43]) # Depth of pair potential / Boltzmann constant (K)
k_ij = np.array([[0.0,   .065, -.0149, -.04838],
                 [.065,   0.0,    0.0, 0.0],
                 [-.0149, 0.0,    0.0, 0.0],
                 [-.04838, 0.0, 0.0, -.00978]])

κ_AB = np.array([0, 0, 0, 0])
ϵ_AB_k = np.array([0, 0, 0, 0])

mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P_sys, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
print(mix_v.φ())