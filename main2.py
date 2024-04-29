# ===================
# Import Dependencies
# ===================
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import loadmat, wavfile

# Define global variables
Himp = None

# ====================================================
# Define Main Functions for Sound Simulation Algorithm
# ====================================================
def lip_displ(p, y, Param):
    # Lip displacement function calculating y(t)
    # Following the method of Philippe Guillemain, Jean Kergomard, and Thierry Voinier (2005)
    
    A = 1 / (Param.fe ** 2 + (2 * np.pi * Param.fl * Param.fe) / (Param.Ql * 2))
    B = 1 / Param.mul
    C = 2 * Param.fe ** 2 - (2 * np.pi * Param.fl) ** 2
    D = ((2 * np.pi * Param.fl * Param.fe) / (Param.Ql * 2) - Param.fe ** 2)

    y.append(max([A * (B * (Param.Pm - p[-1]) + C * y[-1] + D * y[-2]), 0]))

def implicit_func(y, p, u, Param):
    # Implicit function solver for p(t) and u(t), based on iterative updates to y(t)
    # Following the method of Philippe Guillemain, Jean Kergomard, and Thierry Voinier (2005)

    global Himp
    V = past_flow(u)
    K = (2 * Param.b ** 2 * (y[-1] + Param.Ho) ** 2) / Param.rho
    u.append((Param.Pm > V) * (K / 2) * (-Himp[0] + np.sqrt(Himp[0] ** 2 + (4 * abs(V - Param.Pm)) / K)))
    p.append(Himp[0] * u[-1] + V)

def past_flow(u):
    # Function to solve for V, a variable representing the past behavior of u, the acoustic flow velocity
    # Following the method of Philippe Guillemain, Jean Kergomard, and Thierry Voinier (2005)

    global Himp
    uconv = np.flipud(u) # reverse the u array
    return np.dot(Himp[:min(len(u), len(Himp))], uconv[:min(len(u), len(Himp))])

# =========================
# Define Musical Parameters
# =========================
class Param:
    pass

Param.fl = 397    # resonant frequency of vibrating lips (Hz)
Param.mul = 1     # surface density of lips (kg/m^2)
Param.Pm = 4500   # mouthpiece pressure (Pa)
Param.b = 0.01    # mouthpiece width (mm)
Param.Ho = 0.0001 # resting height of the lips (mm)
Param.Ql = 4.3    # quality factor
Param.duree = 2   # duration of the simulation

# ============================
# Define Additional Parameters
# ============================
r = 0.008                                   # instrument input radius (mm)
To = 273.16                                 # temperature conversion factor (Celcius to K)
T = To + 27                                 # temperature of air inside mouthpiece (K)
Co = 331.45 * np.sqrt(T / To)               # speed of sound
Param.rho = 1.2929 * (To / T)                # density
Param.Zc = Param.rho * Co / (np.pi * r ** 2) # characteristic impedance

# ============================
# Loading the Impulse Response
# ============================
data = loadmat('/Users/henryjurney/Downloads/192SeniorProject/impulse_response.mat') # from Dr. Robin Tournemenne over email correspondence (in matlab datatype)
h = data['h'].flatten() # impedance response
fe = data['fe'].flatten()[0] # sampling frequency (8000 Hz)
Param.fe = fe

P = fft(h) # apply a fast fourier transform to the impulse response
L = len(h) # length of the impulse response variable
P2 = np.abs(P / L) # average fft value of h
P1 = P2[:L // 2 + 1] # 
P1[1:-1] = 2 * P1[1:-1]
f = fe * np.arange(L // 2 + 1) / L # define frequency based on sample frequency

# Plot the impulse response
plt.figure()
plt.plot(f, P1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Impulse Response Spectrum')
plt.savefig('/Users/henryjurney/Downloads/192SeniorProject/imp_res.png')
plt.show()

Himp = h[:round(fe * 0.2)] #  impulse response of instrument (h) from 0 to 0.2 times the sample frequency (fe) (rounded)

# ==========================
# Sound Simulation Algorithm
# ==========================
t = np.arange(0, Param.duree + 1 / fe, 1 / fe) # initialize t as array of 0's that spans the duration of the simulation
y = [0, 0] # initial lip displacement set to 0 (lip_displ needs two values of y(t) to function)
p = [0] # initial pressure set to 0
u = [0] # initial acoustic flow velocity set to 0
pext = [] # initialize empty external pressure array

# Main for loop to calculate y(t), p(t) and u(t)
for i in range(len(t)): # len(t) = 16000
    lip_displ(p, y, Param) # calculates y(t) based on 1-dimensional lip displacement model (CITE)
    implicit_func(y, p, u, Param) # solves the implicit equation relating the exciter and the resonator (feedback-loop) for p(t) and u(t)
    pext.append(fe * (p[-1] + u[-1] - (p[-2] + u[-2]))) # creates external pressure array

# Plot the displacement of lips as a function of time y(t)
plt.figure()
plt.plot(t, y[2:])
plt.xlabel('Time (s)')
plt.ylabel('Lip Displacement (mm)')
plt.title('y Verses t Graph')
plt.savefig('/Users/henryjurney/Downloads/192SeniorProject/yvst.png')
plt.show()

# =====================================
# Observing Results and Sound Recording
# =====================================
P = fft(p) # apply a fast fourier transform to the pressure
L = len(p) # length of pressure array
P2 = np.abs(P / L) # average fft value of p
P1 = P2[:L // 2 + 1]
P1[1:-1] = 2 * P1[1:-1]
f = fe * np.arange(L // 2 + 1) / L # redefine the frequency based on this new length (L)

# Plot the fft of Produced Frequencies
plt.figure()
plt.plot(f, P1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('fft Amplitude')
plt.title('fft of Produced Frequencies')
plt.savefig('/Users/henryjurney/Downloads/192SeniorProject/fftson.png')
plt.show()

# Plot p(t) (the waveform)
plt.figure()
plt.plot(t[0:], p[1:])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (kPa)')
plt.title('Pressure vs Time (Waveform)')
plt.savefig('/Users/henryjurney/Downloads/192SeniorProject/son.png')
plt.show()

# Write the waveform p(t) into an audiofile
p2play = p / np.max(np.abs(p)) # normalize the volume
wavfile.write('/Users/henryjurney/Downloads/192SeniorProject/sib.wav', fe, p2play)