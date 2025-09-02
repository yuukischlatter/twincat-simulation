"""
enhanced_simple_viz.py - Enhanced TwinCAT Visualization with FIR Filter
======================================================================

Enhanced version showing:
1. Both signals in one plot (Pure 50Hz + Input with disturbances)
2. FIR filter output in green color
3. Clear comparison of all three signals
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime
from fir_filter import create_configured_filter


def create_enhanced_signals():
    """
    Create signals and apply FIR filter
    """
    print("🌊 Generating Enhanced Signals with FIR Filter...")
    
    # Time array - 100ms, 50µs sampling (like TwinCAT)
    duration = 0.1  # 100ms
    sampling_rate = 1 / 50e-6  # 50µs -> 20kHz
    t = np.arange(0, duration, 1/sampling_rate)
    
    # 1. Pure 50Hz Sine Wave
    pure_50hz = 325.0 * np.sin(2 * np.pi * 50.0 * t)  # 325V peak (~230V RMS)
    
    # 2. Simulated Input Signal (what TwinCAT system sees)
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
    
    # 3. Apply FIR Filter to input signal
    print("🔧 Applying TwinCAT FIR Filter...")
    
    # Create configured FIR filter (exact TwinCAT replica)
    fir_filter = create_configured_filter()
    
    # Process signal through FIR filter in chunks (like TwinCAT does with 8 oversamples)
    fir_output = []
    chunk_size = 8  # TwinCAT processes 8 oversamples at a time
    
    for i in range(0, len(input_signal), chunk_size):
        chunk = input_signal[i:i+chunk_size].tolist()
        if len(chunk) == chunk_size:  # Only process complete chunks
            filtered_sample = fir_filter.Call(chunk)
            # Repeat the filtered sample for each input sample in the chunk
            fir_output.extend([filtered_sample] * chunk_size)
        else:
            # For remaining samples, just append the input (or could interpolate)
            fir_output.extend(chunk)
    
    # Ensure same length as input
    fir_output = np.array(fir_output[:len(input_signal)])
    
    print(f"✅ Generated {len(t)} samples over {duration*1000:.0f}ms")
    print(f"🔧 FIR filter applied with {len(fir_output)} output samples")
    
    return t, pure_50hz, input_signal, fir_output


def create_enhanced_plots():
    """
    Create enhanced plots with all signals in one view
    """
    print("📊 Creating Enhanced TwinCAT Plots...")
    
    # Generate signals
    t, pure_50hz, input_signal, fir_output = create_enhanced_signals()
    
    # Duration for calculations
    duration = 0.1  # 100ms (same as in create_enhanced_signals)
    
    # Convert time to milliseconds for better readability
    t_ms = t * 1000
    
    # Create single plot with all three signals
    fig = go.Figure()
    
    # Plot 1: Pure 50Hz (Blue)
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=pure_50hz,
            mode='lines',
            name='🔵 Pure 50Hz Reference',
            line=dict(color='blue', width=2),
            opacity=0.7
        )
    )
    
    # Plot 2: Input Signal with disturbances (Orange)
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=input_signal,
            mode='lines',
            name='🟠 Input with Harmonics & Drops',
            line=dict(color='orange', width=1.5),
            opacity=0.8
        )
    )
    
    # Plot 3: FIR Filter Output (Green)
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=fir_output,
            mode='lines',
            name='🟢 FIR Filter Output (TwinCAT)',
            line=dict(color='green', width=2),
            opacity=0.9
        )
    )
    
    # Apply TwinCAT Zero Crossing Detection to FIR output
    print("🎯 Applying TwinCAT Zero Crossing Detection...")
    
    # Import and use the TwinCAT zero crossing detector
    from zero_crossing_detector import ZeroCrossingSimulator
    
    # Apply TwinCAT Zero Crossing Detection to FIR output (FIXED VERSION)
    print("🎯 Applying TwinCAT Zero Crossing Detection...")
    
    # Import and use the TwinCAT zero crossing detector
    from zero_crossing_detector import ZeroCrossingSimulator
    
    # Create zero crossing detector with TwinCAT timing (400µs cycles)
    # FIXED: Initialize with proper timing to avoid negative values
    zc_simulator = ZeroCrossingSimulator(task_cycle_time_us=400.0)
    
    # Initialize the detector's timing properly to avoid uint64 overflow
    # Set a reasonable starting time (in ns) to avoid negative calculations
    zc_simulator.current_time_ns = 1000000  # Start at 1ms to avoid negative timing
    zc_simulator.detector.lastZeroCrossingLastCycle = 0  # Initialize properly
    
    # Process FIR output through zero crossing detector
    # TwinCAT processes samples at 400µs intervals
    task_cycle_samples = int(400e-6 * len(fir_output) / duration)  # Samples per 400µs
    
    zero_crossings = []
    for i in range(0, len(fir_output), max(1, task_cycle_samples)):
        if i < len(fir_output):
            # Take voltage sample for this cycle
            voltage_sample = float(fir_output[i])
            
            try:
                # Process through zero crossing detector
                zc_result = zc_simulator.process_voltage_sample(voltage_sample)
                
                if zc_result['zero_crossing_detected']:
                    zc_time_ms = t_ms[i] if i < len(t_ms) else t_ms[-1]
                    zero_crossings.append({
                        'time_ms': zc_time_ms,
                        'voltage_sign': zc_result['voltage_sign_positive'],
                        'frequency_hz': zc_result['frequency_hz']
                    })
            except Exception as e:
                # Skip this sample if there's an error, but continue processing
                print(f"⚠️  Skipped sample at {i}: {e}")
                continue
    
    # Mark zero crossings on the plot
    if zero_crossings:
        zc_times = [zc['time_ms'] for zc in zero_crossings]
        zc_signs = [zc['voltage_sign'] for zc in zero_crossings]
        
        # Separate positive and negative zero crossings
        pos_times = [t for t, sign in zip(zc_times, zc_signs) if sign]
        neg_times = [t for t, sign in zip(zc_times, zc_signs) if not sign]
        
        # Mark positive zero crossings (going from negative to positive)
        if pos_times:
            fig.add_trace(
                go.Scatter(
                    x=pos_times,
                    y=[0] * len(pos_times),
                    mode='markers',
                    name='🔴 Zero Crossings (+)',
                    marker=dict(color='red', size=10, symbol='triangle-up'),
                    hovertemplate='Zero Crossing: %{x:.1f}ms<br>Direction: Positive<extra></extra>'
                )
            )
        
        # Mark negative zero crossings (going from positive to negative)  
        if neg_times:
            fig.add_trace(
                go.Scatter(
                    x=neg_times,
                    y=[0] * len(neg_times),
                    mode='markers',
                    name='🔵 Zero Crossings (-)',
                    marker=dict(color='darkred', size=10, symbol='triangle-down'),
                    hovertemplate='Zero Crossing: %{x:.1f}ms<br>Direction: Negative<extra></extra>'
                )
            )
        
        print(f"🎯 Detected {len(zero_crossings)} zero crossings on FIR output using TwinCAT detector")
        
        # Calculate frequency from TwinCAT zero crossing measurements
        if zero_crossings:
            # Show frequency measurements from TwinCAT detector
            valid_frequencies = [zc['frequency_hz'] for zc in zero_crossings if zc['frequency_hz'] > 0]
            if valid_frequencies:
                avg_freq = np.mean(valid_frequencies)
                print(f"📊 TwinCAT measured frequency: {avg_freq:.3f} Hz (from {len(valid_frequencies)} measurements)")
            
            # Also calculate from timing intervals as verification
            if len(zero_crossings) > 1:
                times = [zc['time_ms'] for zc in zero_crossings]
                intervals = np.diff(times)  # Time between crossings in ms
                # Convert to frequency: half-period intervals -> frequency  
                frequencies_from_timing = 1000.0 / (2 * intervals)  # 1000ms/s, factor 2 for half-period
                avg_freq_timing = np.mean(frequencies_from_timing)
                print(f"📊 Frequency from timing intervals: {avg_freq_timing:.3f} Hz (verification)")
        else:
            print(f"📊 No zero crossings detected - may need longer signal or different parameters")
    
    # Add horizontal zero line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="black",
        line_width=1,
        opacity=0.5,
        annotation_text="Zero Line",
        annotation_position="bottom right"
    )
    
    # Add phase shift annotation for FIR filter
    fir_filter = create_configured_filter()
    phase_shift_50hz = fir_filter.GetPhaseShift(50.0)
    
    # Layout
    fig.update_layout(
        title={
            'text': f'⚡ TwinCAT Signal Analysis - Combined View with Zero Crossings<br>' +
                   f'<sub>🟢 Green = FIR Filtered Signal (where zero crossing detection happens) | ' +
                   f'🔴 Red markers = TwinCAT Zero Crossing Detection Points | ' +
                   f'FIR Phase Shift: {phase_shift_50hz:.1f}° @ 50Hz</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Time [ms]",
        yaxis_title="Voltage [V]",
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        font={'size': 12},
        plot_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', range=[0, 100])
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', range=[-400, 400])
    
    return fig


def create_fir_analysis():
    """
    Create additional FIR filter analysis plot
    """
    print("📈 Creating FIR Filter Analysis...")
    
    # Create configured filter
    fir_filter = create_configured_filter()
    
    # Create subplot: frequency response and phase shift
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            '🔧 FIR Filter Frequency Response',
            '📐 FIR Filter Phase Shift vs Frequency'
        ],
        vertical_spacing=0.15
    )
    
    # Frequency range for analysis
    frequencies = np.linspace(10, 200, 191)  # 10-200 Hz
    phase_shifts = [fir_filter.GetPhaseShift(f) for f in frequencies]
    
    # Plot 1: Simple frequency response (amplitude assumed flat for FIR)
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=np.ones(len(frequencies)),  # FIR filter has flat amplitude response
            mode='lines',
            name='Amplitude Response',
            line=dict(color='blue', width=2),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Mark 50Hz point
    fig.add_trace(
        go.Scatter(
            x=[50],
            y=[1],
            mode='markers',
            name='50Hz Operating Point',
            marker=dict(color='red', size=10, symbol='circle'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Plot 2: Phase shift
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=phase_shifts,
            mode='lines',
            name='Phase Shift',
            line=dict(color='green', width=2),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Mark 50Hz phase shift
    phase_50hz = fir_filter.GetPhaseShift(50.0)
    fig.add_trace(
        go.Scatter(
            x=[50],
            y=[phase_50hz],
            mode='markers',
            name=f'50Hz: {phase_50hz:.1f}°',
            marker=dict(color='red', size=10, symbol='circle'),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Mark harmonics
    harmonics = [150, 250, 350]  # 3rd, 5th, 7th
    harmonic_phases = [fir_filter.GetPhaseShift(h) for h in harmonics]
    
    for harm, phase in zip(harmonics, harmonic_phases):
        fig.add_trace(
            go.Scatter(
                x=[harm],
                y=[phase],
                mode='markers',
                name=f'{harm}Hz: {phase:.1f}°',
                marker=dict(color='orange', size=8, symbol='diamond'),
                showlegend=True if harm == 150 else False
            ),
            row=2, col=1
        )
    
    # Layout
    fig.update_layout(
        title='TwinCAT FIR Filter Characteristics (136th Order)',
        height=700,
        showlegend=True,
        font={'size': 12}
    )
    
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Phase Shift [°]", row=2, col=1)
    
    return fig


def analyze_zero_crossings(t_ms, pure_50hz, fir_output, twincat_crossings):
    """
    Analyze and compare zero crossings from different signals
    """
    print(f"\n🔍 Zero Crossing Analysis:")
    print(f"=" * 50)
    
    # 1. Find zero crossings in pure 50Hz signal
    pure_zc = []
    for i in range(1, len(pure_50hz)):
        if (pure_50hz[i-1] < 0 < pure_50hz[i]) or (pure_50hz[i-1] > 0 > pure_50hz[i]):
            # Linear interpolation for exact crossing time
            if pure_50hz[i] != pure_50hz[i-1]:
                ratio = abs(pure_50hz[i-1]) / abs(pure_50hz[i] - pure_50hz[i-1])
                exact_time_ms = t_ms[i-1] + ratio * (t_ms[i] - t_ms[i-1])
            else:
                exact_time_ms = t_ms[i-1]
            
            pure_zc.append({
                'time_ms': exact_time_ms,
                'direction': 'positive' if pure_50hz[i] > 0 else 'negative'
            })
    
    # 2. Find zero crossings in FIR output
    fir_zc = []
    for i in range(1, len(fir_output)):
        if (fir_output[i-1] < 0 < fir_output[i]) or (fir_output[i-1] > 0 > fir_output[i]):
            # Linear interpolation for exact crossing time
            if fir_output[i] != fir_output[i-1]:
                ratio = abs(fir_output[i-1]) / abs(fir_output[i] - fir_output[i-1])
                exact_time_ms = t_ms[i-1] + ratio * (t_ms[i] - t_ms[i-1])
            else:
                exact_time_ms = t_ms[i-1]
            
            fir_zc.append({
                'time_ms': exact_time_ms,
                'direction': 'positive' if fir_output[i] > 0 else 'negative'
            })
    
    # 3. Print comparison
    print(f"📊 Zero Crossing Counts:")
    print(f"  🔵 Pure 50Hz Signal: {len(pure_zc)} crossings")
    print(f"  🟢 FIR Filtered Signal: {len(fir_zc)} crossings")
    print(f"  🔴 TwinCAT Detected: {len(twincat_crossings)} crossings")
    
    print(f"\n📍 Pure 50Hz Zero Crossings (first 10):")
    for i, zc in enumerate(pure_zc[:10]):
        direction_symbol = "📈" if zc['direction'] == 'positive' else "📉"
        print(f"  {i+1:2}: {zc['time_ms']:6.2f}ms {direction_symbol} ({zc['direction']})")
    
    print(f"\n📍 FIR Filtered Zero Crossings (first 10):")
    for i, zc in enumerate(fir_zc[:10]):
        direction_symbol = "📈" if zc['direction'] == 'positive' else "📉"
        print(f"  {i+1:2}: {zc['time_ms']:6.2f}ms {direction_symbol} ({zc['direction']})")
    
    print(f"\n🎯 TwinCAT Detected Zero Crossings:")
    for i, zc in enumerate(twincat_crossings):
        direction_symbol = "📈" if zc['voltage_sign'] else "📉"
        direction_text = "positive" if zc['voltage_sign'] else "negative"
        freq_text = f"{zc['frequency_hz']:.2f}Hz" if zc['frequency_hz'] > 0 else "no freq"
        print(f"  {i+1:2}: {zc['time_ms']:6.2f}ms {direction_symbol} ({direction_text}) - {freq_text}")
    
    # 4. Calculate phase shift between pure and FIR
    if len(pure_zc) > 0 and len(fir_zc) > 0:
        # Find closest FIR crossing to first pure crossing
        first_pure_time = pure_zc[0]['time_ms']
        closest_fir = min(fir_zc, key=lambda x: abs(x['time_ms'] - first_pure_time))
        phase_shift_ms = closest_fir['time_ms'] - first_pure_time
        
        # Convert to degrees (for 50Hz: 20ms = 360°)
        phase_shift_degrees = (phase_shift_ms / 20.0) * 360.0
        
        print(f"\n🔧 Phase Shift Analysis:")
        print(f"  Pure 50Hz first crossing: {first_pure_time:.2f}ms")
        print(f"  FIR closest crossing: {closest_fir['time_ms']:.2f}ms")
        print(f"  Time shift: {phase_shift_ms:.2f}ms")
        print(f"  Phase shift: {phase_shift_degrees:.1f}°")
        
        # Compare with FIR filter specification
        fir_filter = create_configured_filter()
        expected_phase_shift = fir_filter.GetPhaseShift(50.0)
        print(f"  FIR spec phase shift: {expected_phase_shift:.1f}° @ 50Hz")
        print(f"  Measured vs Expected: {abs(phase_shift_degrees - expected_phase_shift):.1f}° difference")
    
    return pure_zc, fir_zc


def main():
    """
    Main function - enhanced visualization!
    """
    print("🌊 Enhanced TwinCAT Signal Visualization with FIR Filter")
    print("=" * 65)
    
    # Create enhanced plots
    main_fig = create_enhanced_plots()
    fir_fig = create_fir_analysis()
    
    # Save HTML files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_filename = f"enhanced_twincat_signals_{timestamp}.html"
    fir_filename = f"twincat_fir_analysis_{timestamp}.html"
    
    main_fig.write_html(main_filename)
    fir_fig.write_html(fir_filename)
    
    print(f"💾 Enhanced plots saved:")
    print(f"  Main signals: {main_filename}")
    print(f"  FIR analysis: {fir_filename}")
    
    # Try to show in browser
    try:
        main_fig.show()
        print("✅ Main plot opened in browser")
    except Exception as e:
        print(f"⚠️  Browser open failed: {e}")
        print(f"👉 Open {main_filename} manually in your browser")
    
    # Enhanced analysis with zero crossing comparison
    t, pure_50hz, input_signal, fir_output = create_enhanced_signals()
    twincat_crossings = []  # Will be filled if zero crossings detected
    
    # Run the analysis and get TwinCAT crossings from main plot creation
    main_fig = create_enhanced_plots()
    
    # Get TwinCAT crossings from the plot creation (they're created there)
    # We need to extract them or recreate them here for analysis
    # Let's recreate the TwinCAT detection for analysis
    print(f"\n🔧 Recreating TwinCAT Detection for Analysis...")
    
    from zero_crossing_detector import ZeroCrossingSimulator
    zc_sim = ZeroCrossingSimulator(task_cycle_time_us=400.0)
    zc_sim.current_time_ns = 1000000  # Initialize properly
    zc_sim.detector.lastZeroCrossingLastCycle = 0
    
    # Process FIR output
    duration = 0.1  # 100ms
    t_ms = t * 1000
    task_cycle_samples = int(400e-6 * len(fir_output) / duration)
    
    twincat_crossings = []
    for i in range(0, len(fir_output), max(1, task_cycle_samples)):
        if i < len(fir_output):
            try:
                voltage_sample = float(fir_output[i])
                zc_result = zc_sim.process_voltage_sample(voltage_sample)
                if zc_result['zero_crossing_detected']:
                    zc_time_ms = t_ms[i] if i < len(t_ms) else t_ms[-1]
                    twincat_crossings.append({
                        'time_ms': zc_time_ms,
                        'voltage_sign': zc_result['voltage_sign_positive'],
                        'frequency_hz': zc_result['frequency_hz']
                    })
            except:
                continue
    
    # Now run the detailed analysis
    pure_zc, fir_zc = analyze_zero_crossings(t_ms, pure_50hz, fir_output, twincat_crossings)
    
    print(f"\n🎉 Enhanced Visualization Complete!")
    print(f"📂 Files created:")
    print(f"   🔵 {main_filename} - All three signals in one plot")
    print(f"   🔧 {fir_filename} - FIR filter characteristics")
    print(f"👀 This shows:")
    print(f"   🔵 Pure 50Hz reference (blue)")
    print(f"   🟠 Real-world input with harmonics and drops (orange)")
    print(f"   🟢 TwinCAT FIR filter output (green) - smoothed and phase-shifted")
    print(f"   📐 Phase shift effects on zero crossing timing")


if __name__ == "__main__":
    main()