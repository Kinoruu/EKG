import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

from scipy import ndimage
from scipy import signal
from scipy.fft import fft, fftfreq

parser = argparse.ArgumentParser()
parser.add_argument("plik", help="nazwa pliku")
args = parser.parse_args()

plik = args.plik
print("plik {}".format(plik))

with open(plik) as f:
    dane = json.load(f)

print("krok: {} ms/probke".format(dane["krok"]*1000))
print("liczba odczytanych probek: {}".format(len(dane["probki"])))
decimate = signal.decimate(dane["probki"], 10)
print("liczba odczytanych probek: {}".format(len(decimate)))

filtr = signal.firwin2(300, [0, 0.005, 0.01, 1], [1, 1, 0, 0])
wynik = signal.convolve(filtr, decimate, mode='valid')
widmo = fft(wynik)
max_value = max(widmo)
czas = [a*dane["krok"] for a in range(len(widmo))]
widmo[0] = 0  # affects the mean value significantly

for i in range(0, 5000):
    widmo[i] = 0
max_index = np.where(widmo == np.amax(widmo))
for i in range(10000, len(widmo)):
    widmo[i] = 0
max_index = np.where(widmo == np.amax(widmo))

print("Srednia: {} " .format(max_value/1200000))
print("Indeks: {} " .format(max_index))
plt.plot(czas, wynik)
plt.show()
plt.plot(widmo)
plt.show()

# puls[Hz]: ~0.88(3)
# puls[s] : ~1.132
