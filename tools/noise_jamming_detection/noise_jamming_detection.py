import numpy as np
import matplotlib.pyplot as plt

# ANSI escape sequences for colored terminal output
RED = '\033[91m'
RESET = '\033[0m'

# --- CONFIG ---
filename = 'noise_jamming.dat'
sample_rate = 1e6       # Hz (adjust to match your actual sampling rate)
center_freq = 2.45e9    # Center frequency in Hz (for proper labeling)
n_fft = 4096            # FFT size

# --- LOAD DATA ---
raw = np.fromfile(filename, dtype=np.float32)
samples = raw[::2] + 1j * raw[1::2]  # Convert interleaved I/Q to complex

# --- PERFORM FFT ---
spectrum = np.fft.fftshift(np.fft.fft(samples[:n_fft] * np.hanning(n_fft)))
freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1/sample_rate)) + center_freq
power = np.abs(spectrum)**2

# --- DETECT JAMMING ---
mean_power = np.mean(power)
std_power = np.std(power)
threshold = mean_power + 8 * std_power  # adjustable multiplier
jam_indices = np.where(power > threshold)[0]

# --- LOG TO CONSOLE ---
if len(jam_indices) > 0:
    print(f"{RED}WARNING: Potential jamming detected at the following frequencies (Hz):{RESET}")
    for idx in jam_indices:
        print(f"{RED}{freqs[idx]:.2f} Hz (Power: {power[idx]:.2e}){RESET}")
else:
    print("No jamming detected.")

# --- PLOT ---
plt.figure(figsize=(14, 6))
plt.plot(freqs / 1e6, 10 * np.log10(power), label='Spectrum (dB)')

# Highlight jamming points
if len(jam_indices) > 0:
    plt.scatter(freqs[jam_indices] / 1e6,
                10 * np.log10(power[jam_indices]),
                color='red', label='Potential Jamming', zorder=5)

    # Optional: Highlight range with vertical lines or shaded region
    jam_freqs = freqs[jam_indices]
    plt.axvspan(jam_freqs.min() / 1e6, jam_freqs.max() / 1e6,
                color='red', alpha=0.2, label='Jamming Region')
    plt.text((jam_freqs.min() + jam_freqs.max()) / 2 / 1e6,
             np.max(10 * np.log10(power)) - 5,
             'Jamming Identified',
             color='red', ha='center', bbox=dict(facecolor='white', edgecolor='red'))

# Labels and layout
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('Noise Spectrum with Jamming Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
