import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq

# Parameter
sampling_rate = 1 / 50e-6  # 50µs Sampling -> 20 kHz Sampling-Rate
duration = 0.14  # 140ms
t = np.arange(0, duration, 1/sampling_rate)
base_freq = 50  # 50Hz Grundfrequenz

# 1. Reine 50Hz Sinuswelle (blau)
clean_sine = np.sin(2 * np.pi * base_freq * t)

# 2. Sinus mit Einbrüchen und leichten Oberwellen (orange)
distorted_signal = np.sin(2 * np.pi * base_freq * t)

# Realistische Oberwellen für Netzsignal (nicht rechteckig!)
# Hauptsächlich 3. und 5. Harmonische, sehr schwache höhere
harmonics = [3, 5, 7]
harmonic_amplitudes = [0.15, 0.08, 0.04]  # Viel schwächer!

for harm_order, amplitude in zip(harmonics, harmonic_amplitudes):
    freq = base_freq * harm_order
    distorted_signal += amplitude * np.sin(2 * np.pi * freq * t)

# Einbrüche/Störungen - sowohl positive als auch negative!
einbruch_zeiten = [0.02, 0.045, 0.08, 0.105, 0.13]  
for i, einbruch_zeit in enumerate(einbruch_zeiten):
    einbruch_start = int(einbruch_zeit * sampling_rate)
    einbruch_dauer = int(0.003 * sampling_rate)  # 3ms Einbruch
    if einbruch_start + einbruch_dauer < len(distorted_signal):
        # Abwechselnd positive und negative Einbrüche für Symmetrie
        if i % 2 == 0:
            # Spannungseinbruch (nach unten)
            einbruch_faktor = 0.5 + 0.2 * np.random.random()
            distorted_signal[einbruch_start:einbruch_start + einbruch_dauer] *= einbruch_faktor
        else:
            # Spannungsüberhöhung (nach oben) - symmetrisch!
            ueberspannung_faktor = 1.3 + 0.3 * np.random.random()  # 130-160% der Spannung
            distorted_signal[einbruch_start:einbruch_start + einbruch_dauer] *= ueberspannung_faktor

# Mehr Rauschen für realistische "Zackigkeit"
noise = 0.08 * np.random.normal(0, 1, len(t))
distorted_signal += noise

# Plot erstellen
fig = go.Figure()

# Saubere Sinuswelle (blau)
fig.add_trace(go.Scatter(
    x=t, 
    y=clean_sine,
    mode='lines',
    name='Reine 50Hz Sinuswelle',
    line=dict(color='blue', width=2)
))

# Signal mit Einbrüchen (orange/rot)
fig.add_trace(go.Scatter(
    x=t, 
    y=distorted_signal,
    mode='lines', 
    name='Signal mit Einbrüchen und Oberwellen',
    line=dict(color='orange', width=1.5)
))

# Layout anpassen
fig.update_layout(
    title='FIR Filter - 50Hz Signal mit Spannungseinbrüchen',
    xaxis_title='Zeit [s]',
    yaxis_title='Amplitude',
    height=500,
    width=1000,
    showlegend=True,
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[0, duration]
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[-1.5, 1.5]
    )
)

# HTML speichern
fig.write_html("50hz_signal_mit_einbruechen.html")

# FFT Analysis
fft_values = fft(distorted_signal)
fft_freqs = fftfreq(len(distorted_signal), 1/sampling_rate)

# Nur positive Frequenzen bis 1kHz
pos_mask = (fft_freqs > 0) & (fft_freqs <= 1000)
fft_magnitude = np.abs(fft_values[pos_mask])
fft_freqs_pos = fft_freqs[pos_mask]

# FFT Plot
fig_fft = go.Figure()
fig_fft.add_trace(go.Scatter(
    x=fft_freqs_pos, 
    y=fft_magnitude,
    mode='lines+markers',
    name='FFT Spektrum',
    line=dict(color='red', width=2),
    marker=dict(size=4)
))

fig_fft.update_layout(
    title='FFT Spektrum des gestörten Signals',
    xaxis_title='Frequenz [Hz]',
    yaxis_title='Magnitude',
    height=400,
    showlegend=False,
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
)

# HTML speichern  
fig_fft.write_html("50hz_fft_spektrum.html")