import numpy as np
import matplotlib.pyplot as plt
import sys

def load_complex_data(filename):
    raw = np.fromfile(filename, dtype=np.float32)
    return raw[::2] + 1j * raw[1::2]

def match_lengths(x, y):
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]

def normalize(signal):
    return signal / np.max(np.abs(signal))

def main(scan_file, noise_file, sample_rate, noise_scale=0.2):
    scan = load_complex_data(scan_file)
    noise = load_complex_data(noise_file)

    # Match lengths
    scan, noise = match_lengths(scan, noise)

    # Normalize both signals
    scan = normalize(scan)
    noise = normalize(noise) * noise_scale  # Scale noise lower

    # Combine
    combined = scan + noise

    # Windowing
    window = np.hanning(len(combined))
    windowed = combined * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(windowed))

    # dB magnitude (with eps for stability)
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    # Frequency axis (centered)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1/sample_rate)) + center_freq

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(freqs / 1e6, spectrum_db)  # x-axis in MHz
    plt.title("Combined Signal Frequency Spectrum")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    center_freq = 2.4e9  # Hz

    if len(sys.argv) < 4:
        print("Usage: python combine_and_plot.py <scan.dat> <noise.dat> <sample_rate> [noise_scale]")
        sys.exit(1)

    scan_file = sys.argv[1]
    noise_file = sys.argv[2]
    sample_rate = float(sys.argv[3])
    noise_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2  # Default 20% noise

    main(scan_file, noise_file, sample_rate, noise_scale)
