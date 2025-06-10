import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# === Define Chirp Generator ===
def generate_chirp(f_start, f_stop, offsets, duration, samples):
    freq_base = np.linspace(f_start, f_stop, samples, endpoint=False)
    chirps = [freq_base + offset for offset in offsets]
    return np.array(chirps)

# === Radar Setup Function ===
def setup_radar(frequencies, durations, power, prp, pulses, offsets, position, rotation, phase_pattern, fs, rf_gain, noise_figure, bb_gain, resistor):
    tx = Transmitter(
        f=frequencies,
        t=durations,
        tx_power=power,
        prp=prp,
        pulses=pulses,
        f_offset=offsets,
        channels=[dict(location=(0, 0, 0), pulse_phs=phase_pattern)]
    )
    
    rx = Receiver(
        fs=fs,
        noise_figure=noise_figure,
        rf_gain=rf_gain,
        load_resistor=resistor,
        baseband_gain=bb_gain,
        channels=[dict(location=(0, 0, 0))]
    )
    
    return Radar(transmitter=tx, receiver=rx, location=position, rotation=rotation)

# === Setup Victim Radar Only ===
victim_radar = setup_radar(
    frequencies=[60.6e9, 60.4e9],
    durations=[0, 16e-6],
    power=25,
    prp=20e-6,
    pulses=4,
    offsets=np.arange(0, 4) * 90e6,
    position=(0, 0, 0),
    rotation=(0, 0, 0),
    phase_pattern=np.array([180, 0, 0, 0]),
    fs=40e6,
    rf_gain=20,
    noise_figure=2,
    bb_gain=60,
    resistor=500
)

# === Define Targets ===
targets = [
    {"location": (30, 0, 0), "speed": (0, 0, 0), "rcs": 10, "phase": 0},
    {"location": (20, 1, 0), "speed": (-10, 0, 0), "rcs": 10, "phase": 0}
]

# === Perform Radar Simulation Without Interference ===
simulation_data = sim_radar(victim_radar, targets)

# === Extract Data ===
timestamps = simulation_data["timestamp"]
victim_baseband = simulation_data["baseband"]

# === Compute Range FFT ===
def compute_range_fft(baseband, fs):
    N = baseband.shape[-1]
    fft_vals = np.fft.fft(baseband, axis=-1)
    freq_axis = np.fft.fftfreq(N, d=1/fs)
    range_magnitude = 20 * np.log10(np.abs(fft_vals))
    return freq_axis[:N//2], range_magnitude[:, :, :N//2]

freq_axis, range_magnitude = compute_range_fft(
    victim_baseband,
    40e6
)



# === Visualization ===
fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                    subplot_titles=(
                        "Victim Radar Chirps (Targets Only)",
                        "Radar Signal (Complex Baseband)",
                        "Range FFT - Target Position Visualization"
                    ))

# Victim Radar Chirps
victim_chirps = generate_chirp(
    60.6e9,
    60.4e9,
    victim_radar.radar_prop["transmitter"].waveform_prop["f_offset"],
    16e-6,
    victim_radar.sample_prop["samples_per_pulse"]
)

for idx, chirp in enumerate(victim_chirps):
    fig.add_trace(go.Scatter(
        x=timestamps[0, idx, :],
        y=chirp / 1e9,
        mode='lines',
        name='Victim Chirp' if idx == 0 else '',
        line=dict(color='blue'),
        showlegend=(idx == 0),
    ), row=1, col=1)

# Baseband Signal Visualization (Targets only)
num_pulses = victim_radar.radar_prop["transmitter"].waveform_prop["pulses"]

for idx in range(num_pulses):
    fig.add_trace(go.Scatter(
        x=timestamps[0, idx, :],
        y=np.real(victim_baseband[0, idx, :]),
        mode='lines',
        name='Real Part' if idx == 0 else '',
        line=dict(color='green'),
        showlegend=(idx == 0),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=timestamps[0, idx, :],
        y=np.imag(victim_baseband[0, idx, :]),
        mode='lines',
        name='Imag Part' if idx == 0 else '',
        line=dict(color='purple', dash='dot'),
        showlegend=(idx == 0),
    ), row=2, col=1)

# === Range FFT (Frequency Domain - Target Positions) ===
for idx in range(num_pulses):
    fig.add_trace(go.Scatter(
        x=freq_axis / 1e6,
        y=range_magnitude[0, idx, :],
        mode='lines',
        name=f'Pulse {idx + 1}',
        showlegend=True,
    ), row=3, col=1)

# Interactive Layout
fig.update_layout(
    height=1200,
    title="Radar Visualization (Targets Only with Range FFT)",
    xaxis_title="Time (s)",
    yaxis_title="Frequency (GHz)",
    xaxis2_title="Time (s)",
    yaxis2_title="Amplitude (V)",
    xaxis3_title="Frequency (MHz)",
    yaxis3_title="Magnitude (dB)",
    margin=dict(l=20, r=20, t=60, b=20),
    hovermode="x unified"
)

# === Export as Interactive HTML ===
fig.write_html("radar_targets_range_fft.html")

print("Visualization saved as 'radar_targets_range_fft.html'")
