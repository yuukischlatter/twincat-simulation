"""
simple_viz.py - Einfache TwinCAT Visualisierung
===============================================

Einfache, klare Plots:
1. Normale 50Hz Sinuswelle
2. Simuliertes Input Signal (mit Störungen)
3. Das wars - simple!
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime


def create_simple_signals():
    """
    Erstelle einfache Signale für Visualisierung
    """
    print("🌊 Generating Simple Signals...")
    
    # Time array - 100ms, 50µs sampling (wie TwinCAT)
    duration = 0.1  # 100ms
    sampling_rate = 1 / 50e-6  # 50µs -> 20kHz
    t = np.arange(0, duration, 1/sampling_rate)
    
    # 1. Pure 50Hz Sine Wave
    pure_50hz = 325.0 * np.sin(2 * np.pi * 50.0 * t)  # 325V peak (~230V RMS)
    
    # 2. Simulated Input Signal (wie TwinCAT system sieht)
    input_signal = 325.0 * np.sin(2 * np.pi * 50.0 * t)
    
    # Add some realistic harmonics
    input_signal += 0.15 * 325.0 * np.sin(2 * np.pi * 150.0 * t)  # 3rd harmonic
    input_signal += 0.08 * 325.0 * np.sin(2 * np.pi * 250.0 * t)  # 5th harmonic
    
    # Add some voltage drops (Einbrüche)
    drop_times = [0.02, 0.06]  # 20ms, 60ms
    for drop_time in drop_times:
        drop_start = int(drop_time * sampling_rate)
        drop_duration = int(0.003 * sampling_rate)  # 3ms drop
        if drop_start + drop_duration < len(input_signal):
            input_signal[drop_start:drop_start + drop_duration] *= 0.6  # 40% voltage drop
    
    # Add noise
    noise = 8.0 * np.random.normal(0, 1, len(t))
    input_signal += noise
    
    print(f"✅ Generated {len(t)} samples over {duration*1000:.0f}ms")
    
    return t, pure_50hz, input_signal


def create_simple_plots():
    """
    Erstelle einfache, klare Plots
    """
    print("📊 Creating Simple TwinCAT Plots...")
    
    # Generate signals
    t, pure_50hz, input_signal = create_simple_signals()
    
    # Convert time to milliseconds for better readability
    t_ms = t * 1000
    
    # Create subplot: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            '🔵 Pure 50Hz Sine Wave (Perfect Reference)',
            '🟠 Simulated Input Signal (What TwinCAT System Sees)'
        ],
        vertical_spacing=0.15
    )
    
    # Plot 1: Pure 50Hz
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=pure_50hz,
            mode='lines',
            name='Pure 50Hz',
            line=dict(color='blue', width=2),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Plot 2: Input Signal with disturbances
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=input_signal,
            mode='lines',
            name='Input with Harmonics & Drops',
            line=dict(color='orange', width=1.5),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Mark voltage drops
    drop_times_ms = [20, 60]  # 20ms, 60ms
    for drop_time in drop_times_ms:
        fig.add_vline(
            x=drop_time, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Drop @{drop_time}ms",
            annotation_position="top",
            row=2, col=1
        )
    
    # Layout
    fig.update_layout(
        title={
            'text': '⚡ TwinCAT Signal Analysis - Simple View<br>' +
                   '<sub>50Hz Mains Voltage: Pure vs Real-World Input</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=700,
        width=1000,
        showlegend=True,
        font={'size': 12},
        plot_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time [ms]", showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Voltage [V]", showgrid=True, gridcolor='lightgray')
    
    return fig


def analyze_signals():
    """
    Simple signal analysis
    """
    print("🔍 Simple Signal Analysis...")
    
    t, pure_50hz, input_signal = create_simple_signals()
    
    # Basic statistics
    print(f"\n📊 Signal Statistics:")
    print(f"Pure 50Hz:")
    print(f"  Peak: {np.max(pure_50hz):.1f}V")
    print(f"  RMS: {np.sqrt(np.mean(pure_50hz**2)):.1f}V")
    
    print(f"Input Signal:")
    print(f"  Peak: {np.max(input_signal):.1f}V")
    print(f"  RMS: {np.sqrt(np.mean(input_signal**2)):.1f}V")
    print(f"  Min (after drops): {np.min(input_signal):.1f}V")
    
    # Zero crossings count
    pure_zc = np.sum(np.diff(np.sign(pure_50hz)) != 0)
    input_zc = np.sum(np.diff(np.sign(input_signal)) != 0)
    
    print(f"\n🎯 Zero Crossings:")
    print(f"  Pure 50Hz: {pure_zc} crossings")
    print(f"  Input Signal: {input_zc} crossings")
    print(f"  Expected (50Hz, 100ms): {int(50 * 0.1 * 2)} crossings")


def main():
    """
    Main function - super simple!
    """
    print("🌊 Simple TwinCAT Signal Visualization")
    print("=" * 45)
    
    # Create simple plots
    fig = create_simple_plots()
    
    # Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simple_twincat_signals_{timestamp}.html"
    fig.write_html(filename)
    
    print(f"💾 Simple plot saved: {filename}")
    
    # Try to show in browser
    try:
        fig.show()
        print("✅ Plot opened in browser")
    except Exception as e:
        print(f"⚠️  Browser open failed: {e}")
        print(f"👉 Open {filename} manually in your browser")
    
    # Simple analysis
    analyze_signals()
    
    print(f"\n🎉 Simple Visualization Complete!")
    print(f"📂 File created: {filename}")
    print(f"👀 This shows exactly what TwinCAT system processes:")
    print(f"   🔵 Perfect 50Hz reference")
    print(f"   🟠 Real-world input with harmonics and voltage drops")


if __name__ == "__main__":
    main()