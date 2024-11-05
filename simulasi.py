import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameter model
N = 200 + 10 + 50  # Total populasi = 200 (S) + 10 (I) + 50 (R)
I0 = 10            # Individu awal yang terinfeksi
R0 = 50            # Individu awal yang sembuh
S0 = N - I0 - R0   # Individu rentan
beta = 0.3         # Tingkat penularan
gamma = 0.1        # Tingkat pemulihan

# Waktu (hari)
days = 160
t = np.linspace(0, days, days)

# Model SIR
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Kondisi awal
y0 = S0, I0, R0

# Integrasi
result = odeint(sir_model, y0, t, args=(N, beta, gamma))
S, I, R = result.T

# Grafik
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible', color='blue')
plt.plot(t, I, label='Infected', color='red')
plt.plot(t, R, label='Recovered', color='green')
plt.title('Model Penyebaran ISPA menggunakan Model SIR')
plt.xlabel('Hari')
plt.ylabel('Jumlah Individu')
plt.legend()
plt.grid()
plt.show()