"""
signal_generator.py - TwinCAT Integrated Signal Generator
========================================================

Enhanced version of your original signal_generator.py that integrates perfectly
with the complete TwinCAT 1:1 replica system.

Features:
- Your original signal generation logic preserved
- Perfect integration with TwinCAT measurement timing (400µs cycles, 50µs oversamples)
- Realistic 3-phase mains voltage with disturbances
- Coordinated with measurement system for synchronized analysis
- Export functions for TwinCAT visualization system

This maintains your original approach while adding TwinCAT system integration.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
import math
from typing import Tuple, List, Dict, Optional
from datetime import datetime


class TwinCATIntegratedSignGenerator:
    """
    Enhanced signal generator that integrates with TwinCAT measurement system
    
    Based on your original signal_generator.py with TwinCAT timing integration:
    - Maintains your original signal characteristics
    - Adds 3-phase generation for complete system
    - Synchronizes with 400µs measurement cycles
    - Provides 50µs oversample timing
    """
    
    def __init__(self):
        """
        Initialize signal generator with your original parameters plus TwinCAT integration
        """
        # Original parameters from your signal_generator.py
        self.sampling_rate = 1 / 50e-6  # 50µs Sampling -> 20 kHz Sampling-Rate
        self.base_freq = 50  # 50Hz Grundfrequenz
        
        # TwinCAT timing parameters
        self.measurement_cycle_us = 400.0  # 400µs measurement task cycle
        self.oversample_interval_us = 50.0  # 50µs between oversamples
        self.oversamples_per_cycle = 8  # 8 oversamples per measurement cycle
        
        # Your original harmonic configuration
        self.harmonics = [3, 5, 7]
        self.harmonic_amplitudes = [0.15, 0.08, 0.04]  # Realistic levels from your code
        
        # Your original disturbance parameters
        self.einbruch_probability = 0.02  # Probability per cycle
        self.einbruch_duration_ms = 3.0  # 3ms duration as in your original
        
        print(f"🌊 TwinCAT Integrated Signal Generator Initialized")
        print(f"   Base Frequency: {self.base_freq}Hz")
        print(f"   Sampling Rate: {self.sampling_rate/1000:.1f}kHz") 
        print(f"   TwinCAT Timing: {self.measurement_cycle_us}µs cycles, {self.oversample_interval_us}µs oversamples")
    
    def generate_your_original_signal(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate your exact original signal with einbrüche and oberwellen
        
        This preserves your original signal_generator.py logic exactly
        
        Args:
            duration: Signal duration in seconds
            
        Returns:
            Tuple of (time_array, signal_array) - your original format
        """
        # Your original timing setup
        t = np.arange(0, duration, 1/self.sampling_rate)
        
        # 1. Your original pure 50Hz sine wave
        clean_sine = np.sin(2 * np.pi * self.base_freq * t)
        
        # 2. Your original distorted signal with harmonics
        distorted_signal = np.sin(2 * np.pi * self.base_freq * t)
        
        # Add your original harmonics (3rd, 5th, 7th)
        for harm_order, amplitude in zip(self.harmonics, self.harmonic_amplitudes):
            freq = self.base_freq * harm_order
            distorted_signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Your original Einbrüche/Störungen - both positive and negative
        einbruch_zeiten = [0.02, 0.045, 0.08, 0.105, 0.13]  # Your original times
        for i, einbruch_zeit in enumerate(einbruch_zeiten):
            if einbruch_zeit < duration:  # Only add if within duration
                einbruch_start = int(einbruch_zeit * self.sampling_rate)
                einbruch_dauer = int(self.einbruch_duration_ms * 0.001 * self.sampling_rate)
                
                if einbruch_start + einbruch_dauer < len(distorted_signal):
                    # Your original alternating positive/negative einbrüche
                    if i % 2 == 0:
                        # Spannungseinbruch (nach unten)
                        einbruch_faktor = 0.5 + 0.2 * np.random.random()
                        distorted_signal[einbruch_start:einbruch_start + einbruch_dauer] *= einbruch_faktor
                    else:
                        # Spannungsüberhöhung (nach oben) - symmetric!
                        ueberspannung_faktor = 1.3 + 0.3 * np.random.random()
                        distorted_signal[einbruch_start:einbruch_start + einbruch_dauer] *= ueberspannung_faktor
        
        # Your original noise level
        noise = 0.08 * np.random.normal(0, 1, len(t))
        distorted_signal += noise
        
        return t, distorted_signal
    
    def generate_twincat_synchronized_3phase(self, duration_ms: float, 
                                           amplitude_v: float = 325.0) -> Dict:
        """
        Generate 3-phase signals synchronized with TwinCAT measurement timing
        
        Args:
            duration_ms: Duration in milliseconds
            amplitude_v: Peak amplitude in volts (~230V RMS * sqrt(2))
            
        Returns:
            Dict with time points and 3-phase voltages synchronized to TwinCAT cycles
        """
        # Calculate number of measurement cycles
        num_cycles = int(duration_ms / (self.measurement_cycle_us / 1000.0))
        
        # Generate time points for each measurement cycle
        all_cycles_data = []
        
        for cycle_idx in range(num_cycles):
            # Start time of this measurement cycle
            cycle_start_time_s = cycle_idx * (self.measurement_cycle_us / 1e6)
            
            # Generate 8 oversample time points within this cycle
            oversample_times = []
            for sample_idx in range(self.oversamples_per_cycle):
                sample_time_s = cycle_start_time_s + (sample_idx * self.oversample_interval_us / 1e6)
                oversample_times.append(sample_time_s)
            
            oversample_times = np.array(oversample_times)
            
            # Generate 3-phase voltages at these exact time points
            L1_voltages, L2_voltages, L3_voltages = self._generate_3phase_at_times(
                oversample_times, amplitude_v)
            
            # Apply your original disturbances to each phase
            L1_voltages = self._apply_your_disturbances(L1_voltages, oversample_times, cycle_idx)
            L2_voltages = self._apply_your_disturbances(L2_voltages, oversample_times, cycle_idx)
            L3_voltages = self._apply_your_disturbances(L3_voltages, oversample_times, cycle_idx)
            
            # Store cycle data
            cycle_data = {
                'cycle_index': cycle_idx,
                'cycle_start_time_s': cycle_start_time_s,
                'oversample_times_s': oversample_times.tolist(),
                'L1_voltages_v': L1_voltages.tolist(),
                'L2_voltages_v': L2_voltages.tolist(),
                'L3_voltages_v': L3_voltages.tolist(),
                'L2L_voltages_v': (L2_voltages - L1_voltages).tolist()  # Line-to-line for measurement
            }
            
            all_cycles_data.append(cycle_data)
        
        return {
            'duration_ms': duration_ms,
            'num_cycles': num_cycles,
            'measurement_cycle_us': self.measurement_cycle_us,
            'oversample_interval_us': self.oversample_interval_us,
            'cycles_data': all_cycles_data,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_3phase_at_times(self, times: np.ndarray, amplitude: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3-phase voltages with your original characteristics
        """
        omega = 2 * np.pi * self.base_freq
        
        # Base 50Hz sine waves with 120° phase shifts
        L1 = amplitude * np.sin(omega * times)
        L2 = amplitude * np.sin(omega * times - 2*np.pi/3)  # 120° lag
        L3 = amplitude * np.sin(omega * times - 4*np.pi/3)  # 240° lag
        
        # Add your original harmonics to each phase
        for harm_order, harm_amplitude in zip(self.harmonics, self.harmonic_amplitudes):
            harm_omega = omega * harm_order
            L1 += harm_amplitude * amplitude * np.sin(harm_omega * times)
            L2 += harm_amplitude * amplitude * np.sin(harm_omega * times - 2*np.pi/3)
            L3 += harm_amplitude * amplitude * np.sin(harm_omega * times - 4*np.pi/3)
        
        return L1, L2, L3
    
    def _apply_your_disturbances(self, signal: np.ndarray, times: np.ndarray, cycle_idx: int) -> np.ndarray:
        """
        Apply your original disturbance patterns
        """
        disturbed_signal = signal.copy()
        
        # Add occasional voltage drops/spikes based on your original logic
        if np.random.random() < self.einbruch_probability:
            # Apply disturbance to random samples in this cycle
            num_disturbed_samples = min(2, len(signal))  # Max 2 samples per cycle
            disturbed_indices = np.random.choice(len(signal), num_disturbed_samples, replace=False)
            
            for idx in disturbed_indices:
                if cycle_idx % 2 == 0:
                    # Voltage drop (your original range)
                    factor = 0.5 + 0.3 * np.random.random()
                else:
                    # Voltage spike (your original range)
                    factor = 1.2 + 0.4 * np.random.random()
                
                disturbed_signal[idx] *= factor
        
        # Add your original noise level
        noise = 0.08 * np.random.normal(0, 1, len(signal))
        disturbed_signal += noise
        
        return disturbed_signal
    
    def create_your_original_plots(self, duration: float = 0.14, save_html: bool = True) -> Tuple[go.Figure, go.Figure]:
        """
        Create your original plots exactly as in signal_generator.py
        
        Args:
            duration: Duration in seconds (default 140ms from your original)
            save_html: Save plots as HTML files
            
        Returns:
            Tuple of (signal_plot, fft_plot)
        """
        # Generate your original signal
        t, distorted_signal = self.generate_your_original_signal(duration)
        clean_sine = np.sin(2 * np.pi * self.base_freq * t)
        
        # Your original signal plot
        fig_signal = go.Figure()
        
        # Your original traces
        fig_signal.add_trace(go.Scatter(
            x=t, 
            y=clean_sine,
            mode='lines',
            name='Reine 50Hz Sinuswelle',
            line=dict(color='blue', width=2)
        ))
        
        fig_signal.add_trace(go.Scatter(
            x=t, 
            y=distorted_signal,
            mode='lines', 
            name='Signal mit Einbrüchen und Oberwellen',
            line=dict(color='orange', width=1.5)
        ))
        
        # Your original layout
        fig_signal.update_layout(
            title='FIR Filter - 50Hz Signal mit Spannungseinbrüchen',
            xaxis_title='Zeit [s]',
            yaxis_title='Amplitude',
            height=500,
            width=1000,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, duration]),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-1.5, 1.5])
        )
        
        # Your original FFT analysis
        fft_values = fft(distorted_signal)
        fft_freqs = fftfreq(len(distorted_signal), 1/self.sampling_rate)
        
        # Your original FFT filtering
        pos_mask = (fft_freqs > 0) & (fft_freqs <= 1000)
        fft_magnitude = np.abs(fft_values[pos_mask])
        fft_freqs_pos = fft_freqs[pos_mask]
        
        # Your original FFT plot
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
        
        # Save as HTML (your original filenames)
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            signal_filename = f"50hz_signal_mit_einbruechen_{timestamp}.html"
            fft_filename = f"50hz_fft_spektrum_{timestamp}.html"
            
            fig_signal.write_html(signal_filename)
            fig_fft.write_html(fft_filename)
            
            print(f"💾 Your original plots saved:")
            print(f"   Signal: {signal_filename}")
            print(f"   FFT: {fft_filename}")
        
        return fig_signal, fig_fft
    
    def create_twincat_integration_plot(self, duration_ms: float = 100.0, save_html: bool = True) -> go.Figure:
        """
        Create plot showing TwinCAT timing integration
        
        Shows how your original signal integrates with TwinCAT measurement cycles
        """
        # Generate TwinCAT synchronized data
        twincat_data = self.generate_twincat_synchronized_3phase(duration_ms)
        
        # Create comprehensive integration plot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'TwinCAT 3-Phase Voltage Generation (Your Signal + TwinCAT Timing)',
                'Measurement Cycle Synchronization (400µs cycles, 50µs oversamples)', 
                'Line-to-Line Voltage for TwinCAT Measurement System'
            ],
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Collect data from first 20 cycles for visualization
        cycles_to_show = min(20, len(twincat_data['cycles_data']))
        
        all_times_ms = []
        all_L1 = []
        all_L2 = []
        all_L3 = []
        all_L2L = []
        cycle_markers = []
        
        for cycle_data in twincat_data['cycles_data'][:cycles_to_show]:
            times_ms = np.array(cycle_data['oversample_times_s']) * 1000  # Convert to ms
            
            all_times_ms.extend(times_ms)
            all_L1.extend(cycle_data['L1_voltages_v'])
            all_L2.extend(cycle_data['L2_voltages_v'])
            all_L3.extend(cycle_data['L3_voltages_v'])
            all_L2L.extend(cycle_data['L2L_voltages_v'])
            
            # Mark cycle boundaries
            cycle_start_ms = cycle_data['cycle_start_time_s'] * 1000
            cycle_markers.append(cycle_start_ms)
        
        # Plot 1: 3-Phase voltages
        fig.add_trace(go.Scatter(x=all_times_ms, y=all_L1, name='L1 (Red Phase)', 
                                line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=all_times_ms, y=all_L2, name='L2 (Green Phase)', 
                                line=dict(color='green', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=all_times_ms, y=all_L3, name='L3 (Blue Phase)', 
                                line=dict(color='blue', width=1.5)), row=1, col=1)
        
        # Plot 2: Measurement timing
        # Show cycle boundaries
        for i, cycle_start in enumerate(cycle_markers[:10]):  # First 10 cycles
            fig.add_trace(go.Scatter(x=[cycle_start, cycle_start], y=[-400, 400], 
                                    mode='lines', name=f'Cycle {i}' if i < 3 else None,
                                    line=dict(color='gray', width=1, dash='dash'),
                                    showlegend=i < 3), row=2, col=1)
        
        # Show oversamples
        oversample_times_ms = all_times_ms[:40]  # First 5 cycles
        oversample_voltages = all_L2L[:40]
        fig.add_trace(go.Scatter(x=oversample_times_ms, y=oversample_voltages, 
                                mode='markers+lines', name='50µs Oversamples',
                                marker=dict(size=8, color='purple'), 
                                line=dict(color='purple', width=1)), row=2, col=1)
        
        # Plot 3: Line-to-line voltage (what TwinCAT measures)
        fig.add_trace(go.Scatter(x=all_times_ms, y=all_L2L, name='L2-L1 (Measured by TwinCAT)', 
                                line=dict(color='orange', width=2)), row=3, col=1)
        
        # Add markers for your original einbrüche times
        original_einbruch_times_ms = [20, 45, 80]  # From your original code
        for einbruch_time in original_einbruch_times_ms:
            if einbruch_time < duration_ms:
                fig.add_trace(go.Scatter(x=[einbruch_time, einbruch_time], y=[-600, 600],
                                        mode='lines', name='Einbruch' if einbruch_time == 20 else None,
                                        line=dict(color='red', width=2, dash='dot'),
                                        showlegend=einbruch_time == 20), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'TwinCAT Integration: Your Signal Generator + Measurement System<br>' + 
                  f'<sub>Duration: {duration_ms}ms | Cycles: {len(twincat_data["cycles_data"])} | ' +
                  f'Original Harmonics: {self.harmonics} | Einbrüche: Preserved</sub>',
            height=900,
            showlegend=True,
            font={'size': 10}
        )
        
        fig.update_xaxes(title_text="Time [ms]", row=3, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)
        fig.update_yaxes(title_text="L2L Voltage [V]", row=3, col=1)
        
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twincat_signal_integration_{timestamp}.html"
            fig.write_html(filename)
            print(f"💾 TwinCAT integration plot saved: {filename}")
        
        return fig
    
    def export_for_twincat_simulation(self, duration_ms: float = 200.0, filename: Optional[str] = None) -> str:
        """
        Export signal data in format for TwinCAT simulation system
        
        Args:
            duration_ms: Duration in milliseconds
            filename: Optional filename (auto-generated if None)
            
        Returns:
            str: Filename of exported data
        """
        # Generate synchronized 3-phase data
        twincat_data = self.generate_twincat_synchronized_3phase(duration_ms)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twincat_signal_data_{timestamp}.json"
        
        # Save to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(twincat_data, f, indent=2)
        
        print(f"📤 TwinCAT signal data exported: {filename}")
        print(f"   Duration: {duration_ms}ms")
        print(f"   Measurement Cycles: {twincat_data['num_cycles']}")
        print(f"   Total Oversamples: {twincat_data['num_cycles'] * self.oversamples_per_cycle}")
        
        return filename


def main():
    """
    Main function demonstrating integrated signal generation
    """
    print("🌊 TwinCAT Integrated Signal Generator Demo")
    print("=" * 55)
    
    # Create generator
    generator = TwinCATIntegratedSignGenerator()
    
    print(f"\n📊 Generating Your Original Plots...")
    # Create your original plots (140ms duration like your original)
    signal_fig, fft_fig = generator.create_your_original_plots(duration=0.14, save_html=True)
    
    print(f"\n🔧 Creating TwinCAT Integration Analysis...")
    # Create TwinCAT integration plot 
    integration_fig = generator.create_twincat_integration_plot(duration_ms=80.0, save_html=True)
    
    print(f"\n📤 Exporting Data for TwinCAT Simulation...")
    # Export data for TwinCAT simulation system
    export_filename = generator.export_for_twincat_simulation(duration_ms=200.0)
    
    print(f"\n✅ Signal Generation Complete!")
    print(f"   Your original signal characteristics preserved")
    print(f"   TwinCAT timing integration added")
    print(f"   Ready for complete system simulation")
    
    # Demonstrate integration with TwinCAT measurement system
    print(f"\n🚀 Integration with TwinCAT System:")
    print(f"   1. Use your original plots for signal analysis")
    print(f"   2. Use TwinCAT integration plot for system validation")
    print(f"   3. Use exported JSON data with simulation_main.py")
    print(f"   4. All algorithms (FIR, RMS, Zero Crossing) will process your signal!")


if __name__ == "__main__":
    main()