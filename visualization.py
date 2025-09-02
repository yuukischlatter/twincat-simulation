"""
visualization.py - TwinCAT Simulation Analysis & Visualization
=============================================================

TwinCAT Sources: All measurement properties and timing from complete system
- Signal analysis from original signal_generator.py  
- FIR filter response visualization
- Zero crossing detection accuracy
- RMS measurement validation
- Frequency measurement stability
- Complete system performance analysis

Creates comprehensive visualizations to validate the 1:1 TwinCAT replica:
1. Original vs Filtered signals
2. Zero crossing detection with interpolation
3. RMS measurements over time
4. Frequency stability analysis
5. System performance metrics
6. Phase shift compensation effects
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from scipy.fft import fft, fftfreq
from datetime import datetime

# Import TwinCAT simulation components
from simulation_main import TwinCATSystemSimulation, SimulationConfig
from fir_filter import create_configured_filter
from measurement_system import FB_EL3783_LxLy


class TwinCATVisualizationAnalyzer:
    """
    Comprehensive visualization and analysis of TwinCAT simulation results
    
    Creates detailed plots and analysis to validate:
    - FIR filter performance vs original TwinCAT
    - Zero crossing detection accuracy  
    - RMS measurement precision
    - Frequency measurement stability
    - Overall system performance
    """
    
    def __init__(self, results: Dict):
        """
        Initialize analyzer with simulation results
        
        Args:
            results: Complete simulation results from TwinCATSystemSimulation
        """
        self.results = results
        self.config = SimulationConfig(**results['config'])
        self.measurement_data = pd.DataFrame(results['measurement_results'])
        self.signal_data = results['signal_data']
        
        # Extract key measurements
        self.extract_measurement_arrays()
        
        print(f"📊 TwinCAT Visualization Analyzer Initialized")
        print(f"   Total Measurements: {len(self.measurement_data)}")
        print(f"   Analysis Period: {self.config.simulation_duration_ms}ms")
    
    def extract_measurement_arrays(self):
        """
        Extract measurement arrays for efficient plotting
        """
        # Time arrays
        self.time_ms = self.measurement_data['time_ms'].values
        self.time_ns = self.measurement_data['time_ns'].values
        
        # Voltage measurements
        self.l2l_voltage_rms = self.measurement_data['l2l_voltage_rms'].values
        self.frequency_hz = self.measurement_data['frequency_hz'].values
        
        # Zero crossing data
        self.zero_crossing_detected = self.measurement_data['zero_crossing_time_ns'] > 0
        self.zero_crossing_times_ms = (self.measurement_data['zero_crossing_time_ns'] / 1e6).values
        self.voltage_sign = self.measurement_data['voltage_sign_positive'].values
        
        # Filter valid measurements
        self.valid_rms_mask = self.l2l_voltage_rms > 0
        self.valid_freq_mask = self.frequency_hz > 0
        
        print(f"   Valid RMS Measurements: {np.sum(self.valid_rms_mask)}")
        print(f"   Valid Frequency Measurements: {np.sum(self.valid_freq_mask)}")
        print(f"   Zero Crossings Detected: {np.sum(self.zero_crossing_detected)}")
    
    def create_comprehensive_dashboard(self, save_html: bool = True) -> go.Figure:
        """
        Create comprehensive dashboard with all key visualizations
        
        Args:
            save_html: Save dashboard as HTML file
            
        Returns:
            go.Figure: Complete dashboard figure
        """
        print(f"🎨 Creating TwinCAT Comprehensive Dashboard...")
        
        # Create subplot layout
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Original vs Filtered Signal (First 40ms)',
                'Zero Crossing Detection Accuracy',
                'RMS Voltage Measurements', 
                'Frequency Measurement Stability',
                'FFT Spectrum Comparison',
                'System Performance Overview',
                'Phase Shift Analysis',
                'Measurement Statistics'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Original vs Filtered Signal
        self._add_signal_comparison(fig, row=1, col=1)
        
        # Plot 2: Zero Crossing Detection
        self._add_zero_crossing_analysis(fig, row=1, col=2)
        
        # Plot 3: RMS Voltage Measurements
        self._add_rms_analysis(fig, row=2, col=1)
        
        # Plot 4: Frequency Stability
        self._add_frequency_analysis(fig, row=2, col=2)
        
        # Plot 5: FFT Spectrum
        self._add_fft_analysis(fig, row=3, col=1)
        
        # Plot 6: System Performance
        self._add_performance_overview(fig, row=3, col=2)
        
        # Plot 7: Phase Shift Analysis
        self._add_phase_shift_analysis(fig, row=4, col=1)
        
        # Plot 8: Statistics Summary
        self._add_statistics_summary(fig, row=4, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'TwinCAT 1:1 Replica - Complete System Analysis<br>' + 
                       f'<sub>Duration: {self.config.simulation_duration_ms}ms | ' +
                       f'Task Cycle: {self.config.measurement_task_cycle_us}µs | ' +
                       f'Measurements: {len(self.measurement_data)}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=1400,
            showlegend=True,
            font={'size': 10},
            plot_bgcolor='white'
        )
        
        # Save as HTML
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twincat_analysis_dashboard_{timestamp}.html"
            fig.write_html(filename)
            print(f"💾 Dashboard saved as: {filename}")
        
        return fig
    
    def _add_signal_comparison(self, fig: go.Figure, row: int, col: int):
        """
        Add original vs filtered signal comparison
        """
        # Get first few cycles of signal data for detailed view
        if len(self.signal_data) > 0:
            # Take first 40ms worth of data
            cycles_to_show = min(10, len(self.signal_data))
            
            all_times = []
            all_original = []
            all_filtered = []
            
            for cycle_idx in range(cycles_to_show):
                cycle_data = self.signal_data[cycle_idx]
                sample_times_ms = np.array(cycle_data['sample_times']) * 1000  # Convert to ms
                
                # Calculate L2L voltage (L2 - L1)
                l1_voltages = np.array(cycle_data['L1_voltages'])
                l2_voltages = np.array(cycle_data['L2_voltages'])
                l2l_original = l2_voltages - l1_voltages
                
                # Get filtered result from measurement data
                if cycle_idx < len(self.measurement_data):
                    measurement = self.measurement_data.iloc[cycle_idx]
                    if 'lxly_voltage_samples' in measurement and measurement['lxly_voltage_samples']:
                        # Scale back to match original amplitude for comparison
                        filtered_samples = np.array(measurement['lxly_voltage_samples'])
                        # Apply simple scaling to match original for visualization
                        filtered_scaled = filtered_samples * (np.max(np.abs(l2l_original)) / np.max(np.abs(filtered_samples)) if np.max(np.abs(filtered_samples)) > 0 else 1)
                    else:
                        filtered_scaled = l2l_original  # Fallback to original
                else:
                    filtered_scaled = l2l_original
                
                all_times.extend(sample_times_ms)
                all_original.extend(l2l_original)
                all_filtered.extend(filtered_scaled)
            
            # Plot original signal
            fig.add_trace(
                go.Scatter(
                    x=all_times,
                    y=all_original,
                    mode='lines',
                    name='Original L2L',
                    line=dict(color='orange', width=2),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Plot filtered signal  
            fig.add_trace(
                go.Scatter(
                    x=all_times,
                    y=all_filtered,
                    mode='lines',
                    name='FIR Filtered',
                    line=dict(color='blue', width=1.5),
                    showlegend=True
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Time [ms]", row=row, col=col)
        fig.update_yaxes(title_text="Voltage [V]", row=row, col=col)
    
    def _add_zero_crossing_analysis(self, fig: go.Figure, row: int, col: int):
        """
        Add zero crossing detection analysis
        """
        # Filter zero crossings within first 100ms for detailed view
        zc_mask = (self.zero_crossing_times_ms > 0) & (self.zero_crossing_times_ms < 100)
        zc_times = self.zero_crossing_times_ms[zc_mask]
        zc_signs = self.voltage_sign[zc_mask]
        
        if len(zc_times) > 0:
            # Plot zero crossings with different colors for positive/negative
            pos_mask = zc_signs
            neg_mask = ~zc_signs
            
            if np.any(pos_mask):
                fig.add_trace(
                    go.Scatter(
                        x=zc_times[pos_mask],
                        y=np.ones(np.sum(pos_mask)),
                        mode='markers',
                        name='Positive ZC',
                        marker=dict(color='red', size=8, symbol='triangle-up'),
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            if np.any(neg_mask):
                fig.add_trace(
                    go.Scatter(
                        x=zc_times[neg_mask],
                        y=-np.ones(np.sum(neg_mask)),
                        mode='markers',
                        name='Negative ZC',
                        marker=dict(color='blue', size=8, symbol='triangle-down'),
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            # Calculate expected zero crossings for 50Hz
            expected_interval_ms = 1000.0 / (2 * 50.0)  # Half period = 10ms
            expected_times = np.arange(0, 100, expected_interval_ms)
            
            fig.add_trace(
                go.Scatter(
                    x=expected_times,
                    y=np.zeros(len(expected_times)),
                    mode='markers',
                    name='Expected (50Hz)',
                    marker=dict(color='green', size=4, symbol='x'),
                    showlegend=True
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Time [ms]", row=row, col=col)
        fig.update_yaxes(title_text="Polarity", row=row, col=col)
    
    def _add_rms_analysis(self, fig: go.Figure, row: int, col: int):
        """
        Add RMS voltage measurement analysis
        """
        valid_rms = self.l2l_voltage_rms[self.valid_rms_mask]
        valid_times = self.time_ms[self.valid_rms_mask]
        
        if len(valid_rms) > 0:
            # Plot RMS measurements
            fig.add_trace(
                go.Scatter(
                    x=valid_times,
                    y=valid_rms,
                    mode='lines+markers',
                    name='Measured RMS',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Expected value line
            fig.add_trace(
                go.Scatter(
                    x=[valid_times[0], valid_times[-1]],
                    y=[self.config.expected_l2l_voltage_rms, self.config.expected_l2l_voltage_rms],
                    mode='lines',
                    name=f'Expected ({self.config.expected_l2l_voltage_rms}V)',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Tolerance bands
            tolerance_v = self.config.expected_l2l_voltage_rms * (self.config.voltage_tolerance_percent / 100)
            fig.add_trace(
                go.Scatter(
                    x=list(valid_times) + list(valid_times[::-1]),
                    y=list(np.ones(len(valid_times)) * (self.config.expected_l2l_voltage_rms + tolerance_v)) + 
                      list(np.ones(len(valid_times)) * (self.config.expected_l2l_voltage_rms - tolerance_v))[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'±{self.config.voltage_tolerance_percent}% Tolerance',
                    showlegend=True
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Time [ms]", row=row, col=col)
        fig.update_yaxes(title_text="RMS Voltage [V]", row=row, col=col)
    
    def _add_frequency_analysis(self, fig: go.Figure, row: int, col: int):
        """
        Add frequency measurement stability analysis
        """
        valid_freq = self.frequency_hz[self.valid_freq_mask]
        valid_times = self.time_ms[self.valid_freq_mask]
        
        if len(valid_freq) > 0:
            # Plot frequency measurements
            fig.add_trace(
                go.Scatter(
                    x=valid_times,
                    y=valid_freq,
                    mode='lines+markers',
                    name='Measured Frequency',
                    line=dict(color='darkgreen', width=2),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Expected frequency line
            fig.add_trace(
                go.Scatter(
                    x=[valid_times[0], valid_times[-1]],
                    y=[self.config.expected_frequency_hz, self.config.expected_frequency_hz],
                    mode='lines',
                    name=f'Expected ({self.config.expected_frequency_hz}Hz)',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Tolerance bands
            tolerance_hz = self.config.frequency_tolerance_hz
            fig.add_trace(
                go.Scatter(
                    x=list(valid_times) + list(valid_times[::-1]),
                    y=list(np.ones(len(valid_times)) * (self.config.expected_frequency_hz + tolerance_hz)) + 
                      list(np.ones(len(valid_times)) * (self.config.expected_frequency_hz - tolerance_hz))[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'±{tolerance_hz}Hz Tolerance',
                    showlegend=True
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Time [ms]", row=row, col=col)
        fig.update_yaxes(title_text="Frequency [Hz]", row=row, col=col)
    
    def _add_fft_analysis(self, fig: go.Figure, row: int, col: int):
        """
        Add FFT spectrum analysis
        """
        if len(self.signal_data) > 0:
            # Reconstruct signal for FFT analysis
            all_l2l_samples = []
            sample_rate_hz = 1.0 / (self.config.oversample_interval_us * 1e-6)  # 20kHz
            
            for cycle_data in self.signal_data[:50]:  # Use first 50 cycles
                l1 = np.array(cycle_data['L1_voltages'])
                l2 = np.array(cycle_data['L2_voltages'])
                l2l = l2 - l1
                all_l2l_samples.extend(l2l)
            
            if len(all_l2l_samples) > 64:  # Minimum for meaningful FFT
                # Compute FFT
                fft_values = fft(all_l2l_samples)
                fft_freqs = fftfreq(len(all_l2l_samples), 1/sample_rate_hz)
                
                # Get positive frequencies up to 1kHz
                pos_mask = (fft_freqs > 0) & (fft_freqs <= 1000)
                fft_magnitude = np.abs(fft_values[pos_mask])
                fft_freqs_pos = fft_freqs[pos_mask]
                
                # Normalize magnitude
                fft_magnitude_norm = fft_magnitude / np.max(fft_magnitude)
                
                fig.add_trace(
                    go.Scatter(
                        x=fft_freqs_pos,
                        y=20 * np.log10(fft_magnitude_norm + 1e-10),  # dB scale
                        mode='lines',
                        name='Signal Spectrum',
                        line=dict(color='navy', width=1.5),
                        showlegend=True
                    ),
                    row=row, col=col
                )
                
                # Mark fundamental and harmonics
                harmonics = [50, 150, 250, 350]  # 1st, 3rd, 5th, 7th
                for harm in harmonics:
                    if harm <= 1000:
                        fig.add_trace(
                            go.Scatter(
                                x=[harm, harm],
                                y=[-60, 0],
                                mode='lines',
                                name=f'{harm}Hz' if harm == 50 else f'{harm}Hz ({harm//50}rd)',
                                line=dict(color='red' if harm == 50 else 'orange', 
                                         width=2 if harm == 50 else 1, 
                                         dash='solid' if harm == 50 else 'dash'),
                                showlegend=True if harm in [50, 150] else False
                            ),
                            row=row, col=col
                        )
        
        fig.update_xaxes(title_text="Frequency [Hz]", row=row, col=col)
        fig.update_yaxes(title_text="Magnitude [dB]", row=row, col=col)
    
    def _add_performance_overview(self, fig: go.Figure, row: int, col: int):
        """
        Add system performance overview
        """
        # Calculate performance metrics
        total_cycles = len(self.measurement_data)
        rms_success_rate = np.sum(self.valid_rms_mask) / total_cycles * 100
        freq_success_rate = np.sum(self.valid_freq_mask) / total_cycles * 100
        zc_success_rate = np.sum(self.zero_crossing_detected) / total_cycles * 100
        
        # Performance bars
        categories = ['RMS Measurements', 'Frequency Measurements', 'Zero Crossings']
        success_rates = [rms_success_rate, freq_success_rate, zc_success_rate]
        colors = ['purple', 'darkgreen', 'orange']
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=success_rates,
                name='Success Rate',
                marker=dict(color=colors),
                text=[f'{rate:.1f}%' for rate in success_rates],
                textposition='auto',
                showlegend=True
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Measurement Type", row=row, col=col)
        fig.update_yaxes(title_text="Success Rate [%]", range=[0, 105], row=row, col=col)
    
    def _add_phase_shift_analysis(self, fig: go.Figure, row: int, col: int):
        """
        Add FIR filter phase shift analysis
        """
        # Create FIR filter for analysis
        fir_filter = create_configured_filter()
        
        # Calculate phase shift over frequency range
        frequencies = np.linspace(10, 100, 91)  # 10-100 Hz
        phase_shifts = [fir_filter.GetPhaseShift(f) for f in frequencies]
        
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase_shifts,
                mode='lines',
                name='FIR Phase Shift',
                line=dict(color='red', width=2),
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Mark 50Hz point
        phase_50hz = fir_filter.GetPhaseShift(50.0)
        fig.add_trace(
            go.Scatter(
                x=[50],
                y=[phase_50hz],
                mode='markers',
                name=f'50Hz: {phase_50hz:.1f}°',
                marker=dict(color='blue', size=10, symbol='circle'),
                showlegend=True
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Frequency [Hz]", row=row, col=col)
        fig.update_yaxes(title_text="Phase Shift [°]", row=row, col=col)
    
    def _add_statistics_summary(self, fig: go.Figure, row: int, col: int):
        """
        Add measurement statistics summary
        """
        # Calculate key statistics
        if np.any(self.valid_rms_mask):
            rms_mean = np.mean(self.l2l_voltage_rms[self.valid_rms_mask])
            rms_std = np.std(self.l2l_voltage_rms[self.valid_rms_mask])
            rms_accuracy = abs(rms_mean - self.config.expected_l2l_voltage_rms) / self.config.expected_l2l_voltage_rms * 100
        else:
            rms_mean = rms_std = rms_accuracy = 0
            
        if np.any(self.valid_freq_mask):
            freq_mean = np.mean(self.frequency_hz[self.valid_freq_mask])
            freq_std = np.std(self.frequency_hz[self.valid_freq_mask])
            freq_accuracy = abs(freq_mean - self.config.expected_frequency_hz)
        else:
            freq_mean = freq_std = freq_accuracy = 0
        
        # Create text summary
        summary_text = f"""<b>TwinCAT Measurement Statistics</b><br><br>
        <b>Voltage (L2L):</b><br>
        Mean: {rms_mean:.1f}V<br>
        Std: {rms_std:.1f}V<br>
        Accuracy: {rms_accuracy:.2f}% error<br><br>
        
        <b>Frequency:</b><br>
        Mean: {freq_mean:.4f}Hz<br>
        Std: {freq_std:.4f}Hz<br>
        Accuracy: {freq_accuracy:.4f}Hz error<br><br>
        
        <b>System Performance:</b><br>
        Total Cycles: {len(self.measurement_data)}<br>
        RMS Success: {np.sum(self.valid_rms_mask)}/{len(self.measurement_data)}<br>
        Freq Success: {np.sum(self.valid_freq_mask)}/{len(self.measurement_data)}<br>
        ZC Detected: {np.sum(self.zero_crossing_detected)}<br><br>
        
        <b>Validation:</b><br>
        Voltage: {'✅ PASS' if rms_accuracy <= self.config.voltage_tolerance_percent else '❌ FAIL'}<br>
        Frequency: {'✅ PASS' if freq_accuracy <= self.config.frequency_tolerance_hz else '❌ FAIL'}
        """
        
        # Add text annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            text=summary_text,
            showarrow=False,
            font=dict(size=10, family='monospace'),
            align='left',
            xref=f"x{row*2 + col - 1}" if row > 1 else f"x{col}",
            yref=f"y{row*2 + col - 1}" if row > 1 else f"y{col}",
            xanchor='center',
            yanchor='middle',
            bgcolor='rgba(240,240,240,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
        
        # Hide axes for text plot
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
    
    def generate_performance_report(self) -> str:
        """
        Generate detailed performance report
        
        Returns:
            str: Formatted performance report
        """
        analysis = self.results.get('analysis', {})
        
        report = f"""
TwinCAT 1:1 Replica - Performance Validation Report
==================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SYSTEM CONFIGURATION:
- Simulation Duration: {self.config.simulation_duration_ms}ms
- Task Cycle: {self.config.measurement_task_cycle_us}µs  
- Oversamples: {self.config.oversamples_per_cycle} every {self.config.oversample_interval_us}µs
- Total Measurement Cycles: {len(self.measurement_data)}

MEASUREMENT PERFORMANCE:
- RMS Measurements: {np.sum(self.valid_rms_mask)}/{len(self.measurement_data)} ({np.sum(self.valid_rms_mask)/len(self.measurement_data)*100:.1f}%)
- Frequency Measurements: {np.sum(self.valid_freq_mask)}/{len(self.measurement_data)} ({np.sum(self.valid_freq_mask)/len(self.measurement_data)*100:.1f}%)
- Zero Crossings Detected: {np.sum(self.zero_crossing_detected)} ({np.sum(self.zero_crossing_detected)/len(self.measurement_data)*100:.1f}%)

ACCURACY VALIDATION:
"""
        
        if 'voltage' in analysis:
            volt = analysis['voltage']
            report += f"""
Voltage Measurement:
- Measured: {volt['mean_v']:.2f} ± {volt['std_v']:.2f}V
- Expected: {self.config.expected_l2l_voltage_rms}V
- Error: {volt['accuracy_error_percent']:.3f}%
- Status: {'✅ PASS' if volt['within_tolerance'] else '❌ FAIL'} (tolerance: ±{self.config.voltage_tolerance_percent}%)
"""
        
        if 'frequency' in analysis:
            freq = analysis['frequency']
            report += f"""
Frequency Measurement:
- Measured: {freq['mean_hz']:.6f} ± {freq['std_hz']:.6f}Hz
- Expected: {self.config.expected_frequency_hz}Hz
- Error: {freq['accuracy_error_hz']:.6f}Hz
- Status: {'✅ PASS' if freq['within_tolerance'] else '❌ FAIL'} (tolerance: ±{self.config.frequency_tolerance_hz}Hz)
"""
        
        if 'zero_crossings' in analysis:
            zc = analysis['zero_crossings']
            report += f"""
Zero Crossing Accuracy:
- Average Interval: {zc['avg_interval_ms']:.4f}ms
- Expected Interval: {zc['expected_interval_ms']:.4f}ms
- Timing Error: {zc['timing_accuracy_ms']:.4f}ms
- Sub-sample Precision: ±{abs(zc['timing_accuracy_ms']*1000):.1f}µs
"""
        
        if 'fir_filter' in analysis:
            fir = analysis['fir_filter']
            report += f"""
FIR Filter Performance:
- Configuration: {'✅ SUCCESS' if fir['configured'] else '❌ FAILED'}
- Phase Shift @ 50Hz: {fir['phase_shift_50hz_deg']:.2f}°
- Filter Order: 136 (137 coefficients)
- Type: Linear phase FIR with symmetric coefficients
"""
        
        report += f"""
TWINCAT REPLICA VALIDATION:
{'='*50}
"""
        
        # Overall validation status
        voltage_pass = analysis.get('voltage', {}).get('within_tolerance', False) if 'voltage' in analysis else False
        freq_pass = analysis.get('frequency', {}).get('within_tolerance', False) if 'frequency' in analysis else False
        filter_pass = analysis.get('fir_filter', {}).get('configured', False) if 'fir_filter' in analysis else False
        
        if voltage_pass and freq_pass and filter_pass:
            report += "🎉 OVERALL STATUS: ✅ ALL SYSTEMS VALIDATED\n"
            report += "   - TwinCAT algorithms replicated successfully\n"
            report += "   - Measurement accuracy within specifications\n"
            report += "   - 1:1 replica validation COMPLETE\n"
        else:
            report += "⚠️  OVERALL STATUS: ❌ VALIDATION ISSUES DETECTED\n"
            if not voltage_pass:
                report += "   - Voltage measurement accuracy outside tolerance\n"
            if not freq_pass:
                report += "   - Frequency measurement accuracy outside tolerance\n"
            if not filter_pass:
                report += "   - FIR filter configuration failed\n"
        
        return report


def create_comparison_with_original_signal(results: Dict, save_html: bool = True) -> go.Figure:
    """
    Create detailed comparison with original signal generator
    
    Shows the complete pipeline from signal generation through measurement
    """
    print(f"📈 Creating Signal Pipeline Comparison...")
    
    config = SimulationConfig(**results['config'])
    signal_data = results['signal_data'][:10]  # First 10 cycles
    measurement_data = results['measurement_results'][:10]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Original 3-Phase Voltages (L1, L2, L3)',
            'Line-to-Line Voltage (L2-L1) vs Measurement System Output', 
            'Measurement Results: RMS, Frequency, Zero Crossings'
        ],
        vertical_spacing=0.1
    )
    
    # Reconstruct time series
    all_times = []
    all_L1 = []
    all_L2 = []
    all_L3 = []
    all_L2L_original = []
    all_L2L_measured = []
    
    for i, cycle_data in enumerate(signal_data):
        times_ms = np.array(cycle_data['sample_times']) * 1000
        L1 = np.array(cycle_data['L1_voltages'])
        L2 = np.array(cycle_data['L2_voltages'])
        L3 = np.array(cycle_data['L3_voltages'])
        L2L_orig = L2 - L1
        
        # Get measured data
        if i < len(measurement_data):
            measured_samples = measurement_data[i].get('lxly_voltage_samples', L2L_orig.tolist())
            L2L_meas = np.array(measured_samples)
        else:
            L2L_meas = L2L_orig
        
        all_times.extend(times_ms)
        all_L1.extend(L1)
        all_L2.extend(L2)
        all_L3.extend(L3)
        all_L2L_original.extend(L2L_orig)
        all_L2L_measured.extend(L2L_meas)
    
    # Plot 1: 3-Phase voltages
    fig.add_trace(go.Scatter(x=all_times, y=all_L1, name='L1', line=dict(color='red', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=all_times, y=all_L2, name='L2', line=dict(color='green', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=all_times, y=all_L3, name='L3', line=dict(color='blue', width=1.5)), row=1, col=1)
    
    # Plot 2: L2L comparison
    fig.add_trace(go.Scatter(x=all_times, y=all_L2L_original, name='L2L Original', line=dict(color='orange', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=all_times, y=all_L2L_measured, name='L2L Measured', line=dict(color='purple', width=1.5)), row=2, col=1)
    
    # Plot 3: Measurement results
    meas_df = pd.DataFrame(measurement_data)
    valid_mask = (meas_df['l2l_voltage_rms'] > 0) | (meas_df['frequency_hz'] > 0)
    
    if np.any(valid_mask):
        valid_times = meas_df['time_ms'][valid_mask]
        valid_rms = meas_df['l2l_voltage_rms'][valid_mask]
        valid_freq = meas_df['frequency_hz'][valid_mask] * 10  # Scale for visibility
        
        fig.add_trace(go.Scatter(x=valid_times, y=valid_rms, name='RMS [V]', 
                                line=dict(color='red', width=2), marker=dict(size=6)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_times, y=valid_freq, name='Frequency [Hz×10]', 
                                line=dict(color='green', width=2), marker=dict(size=6)), row=3, col=1)
    
    # Mark zero crossings
    zc_data = [(m['time_ms'], m['zero_crossing_time_ns']) for m in measurement_data if m['zero_crossing_time_ns'] > 0]
    if zc_data:
        zc_times, zc_ns = zip(*zc_data)
        fig.add_trace(go.Scatter(x=list(zc_times), y=[500]*len(zc_times), mode='markers',
                                name='Zero Crossings', marker=dict(color='black', size=8, symbol='x')), row=3, col=1)
    
    fig.update_layout(
        title='TwinCAT Signal Processing Pipeline - Original vs Measured',
        height=900,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time [ms]", row=3, col=1)
    
    if save_html:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"twincat_signal_pipeline_{timestamp}.html"
        fig.write_html(filename)
        print(f"💾 Pipeline comparison saved as: {filename}")
    
    return fig


def main():
    """
    Main visualization function
    """
    print("🎨 TwinCAT Visualization & Analysis")
    print("=" * 50)
    
    # Load simulation results (run simulation if needed)
    try:
        # Try to load existing results
        with open('twincat_simulation_results.json', 'r') as f:
            results = json.load(f)
        print("📁 Loaded existing simulation results")
    except FileNotFoundError:
        # Run new simulation
        print("🚀 Running new TwinCAT simulation...")
        from simulation_main import TwinCATSystemSimulation, SimulationConfig
        
        config = SimulationConfig(simulation_duration_ms=100.0)  # Shorter for visualization
        simulation = TwinCATSystemSimulation(config)
        results = simulation.run_simulation()
    
    # Create analyzer
    analyzer = TwinCATVisualizationAnalyzer(results)
    
    # Generate comprehensive dashboard
    dashboard = analyzer.create_comprehensive_dashboard(save_html=True)
    
    # Create signal pipeline comparison
    pipeline_fig = create_comparison_with_original_signal(results, save_html=True)
    
    # Generate performance report
    report = analyzer.generate_performance_report()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"twincat_performance_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📊 Analysis Complete!")
    print(f"💾 Files generated:")
    print(f"   - HTML Dashboard: twincat_analysis_dashboard_*.html")
    print(f"   - Pipeline Analysis: twincat_signal_pipeline_*.html") 
    print(f"   - Performance Report: {report_filename}")
    
    # Print summary
    print(f"\n{report}")


if __name__ == "__main__":
    main()