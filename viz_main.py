"""
viz_main.py - TwinCAT FIR Filter Workflow Visualization
======================================================

Shows the exact TwinCAT FIR filter processing workflow:
Step 1: Signal Generation (Perfect 50Hz vs Your Generated Signal)
Step 2: FIR Filter Processing 
Step 3: Phase Shift Compensation
Step 4: Zero Crossing Detection with Timing Accuracy
Step 5: Final Frequency Measurement

This matches the exact workflow in measurement_system.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from typing import Dict, List, Tuple
from datetime import datetime

# Import TwinCAT components
from fir_filter import create_configured_filter
from zero_crossing_detector import ZeroCrossing
from signal_generator import TwinCATIntegratedSignGenerator


class TwinCATFIRWorkflowViz:
    """
    Complete TwinCAT FIR filter workflow visualization
    
    Shows exact processing chain: Signal → FIR → Phase Compensation → Zero Crossing → Frequency
    """
    
    def __init__(self):
        self.signal_generator = TwinCATIntegratedSignGenerator()
        self.fir_filter = create_configured_filter()
        self.zero_crossing = ZeroCrossing()
        
        print("🎯 TwinCAT FIR Filter Workflow Visualization Ready")
        print(f"   FIR Filter: 136th order, {self.fir_filter.phaseShiftP1}° @ {self.fir_filter.frequencyP1}Hz")
        print(f"   Phase @ 50Hz: {self.fir_filter.GetPhaseShift(50.0):.2f}°")
    
    def step1_generate_signals(self, duration_ms: float = 100.0) -> Dict:
        """
        Step 1: Generate Perfect 50Hz + Your Signal with disturbances
        """
        print(f"\n📊 Step 1: Signal Generation ({duration_ms}ms)")
        
        # Create time array for continuous signals (high resolution for smooth plotting)
        sample_rate = 20000  # 20kHz for smooth visualization
        duration_s = duration_ms / 1000.0
        time_s = np.linspace(0, duration_s, int(sample_rate * duration_s))
        
        # Perfect 50Hz reference signal
        perfect_50hz = 325.0 * np.sin(2 * np.pi * 50.0 * time_s)
        
        # Your generated signal using the signal generator
        # Get your original signal characteristics
        your_time, your_signal = self.signal_generator.generate_your_original_signal(duration_s)
        
        # Scale your signal to match voltage levels (your signal is normalized, scale to 325V peak)
        your_signal_scaled = your_signal * 325.0
        
        # Interpolate to match our time array for consistent plotting
        your_signal_interp = np.interp(time_s, your_time, your_signal_scaled)
        #your_signal_interp = perfect_50hz
        
        print(f"   ✅ Perfect 50Hz: {len(perfect_50hz)} samples")
        print(f"   ✅ Your Signal: {len(your_signal_interp)} samples (with einbrüche & harmonics)")
        
        return {
            'time_s': time_s,
            'perfect_50hz': perfect_50hz,
            'your_signal': your_signal_interp,
            'duration_ms': duration_ms,
            'sample_rate': sample_rate
        }
    
    def step2_apply_fir_filter(self, signals: Dict) -> Dict:
        """
        Step 2: Apply FIR filter to your generated signal
        
        Process signal in 8-sample chunks like TwinCAT does
        """
        print(f"\n🔧 Step 2: FIR Filter Processing")
        
        time_s = signals['time_s']
        your_signal = signals['your_signal']
        
        # Process signal in 8-sample chunks (TwinCAT oversampling pattern)
        # Convert to samples every 50µs (like TwinCAT oversampling)
        oversample_rate = 20000  # 50µs = 20kHz
        oversample_indices = np.arange(0, len(your_signal), len(your_signal) // int(len(your_signal) * 50e-6 * oversample_rate))
        oversample_indices = oversample_indices[oversample_indices < len(your_signal)]
        
        filtered_times = []
        filtered_values = []
        
        # Process in groups of 8 samples (like TwinCAT)
        for i in range(0, len(oversample_indices) - 7, 8):
            # Get 8 consecutive samples
            sample_indices = oversample_indices[i:i+8]
            if len(sample_indices) == 8:
                voltage_samples = [your_signal[idx] for idx in sample_indices]
                
                # Apply FIR filter (returns single filtered value for 8 input samples)
                try:
                    filtered_value = self.fir_filter.Call(voltage_samples)
                    
                    # Use the middle time of the 8 samples
                    filtered_time = time_s[sample_indices[3]]  # Middle sample time
                    
                    filtered_times.append(filtered_time)
                    filtered_values.append(filtered_value)
                    
                except Exception as e:
                    continue
        
        # Convert to numpy arrays and interpolate for smooth plotting
        filtered_times = np.array(filtered_times)
        filtered_values = np.array(filtered_values)
        
        # Interpolate filtered signal to match original time array
        fir_output_interp = np.interp(time_s, filtered_times, filtered_values)
        
        print(f"   ✅ FIR Filter Applied: {len(filtered_times)} filtered points")
        print(f"   ✅ Interpolated for plotting: {len(fir_output_interp)} samples")
        
        return {
            **signals,  # Keep original signals
            'fir_output': fir_output_interp,
            'fir_times': filtered_times,
            'fir_values': filtered_values
        }
    
    def step3_phase_compensation(self, signals: Dict, target_frequency: float = 50.0) -> Dict:
        """
        Step 3: Apply phase shift compensation (like TwinCAT measurement_system.py)
        """
        print(f"\n⚡ Step 3: Phase Shift Compensation @ {target_frequency}Hz")
        
        # Calculate phase delay (from measurement_system.py logic)
        phase_shift_deg = self.fir_filter.GetPhaseShift(target_frequency)
        delay_seconds = -(phase_shift_deg / 360.0) * (1.0 / target_frequency)
        delay_ms = delay_seconds * 1000
        
        print(f"   📐 Phase Shift: {phase_shift_deg:.2f}°")
        print(f"   ⏰ Time Delay: {delay_ms:.2f}ms")
        
        # Apply time shift compensation to FIR output
        time_s = signals['time_s']
        fir_output = signals['fir_output']
        
        # Shift time array forward by delay amount (compensate for delay)
        time_shifted = time_s + delay_seconds  # FIXED: Shift forward to compensate
        
        # Interpolate FIR output to compensated time points
        # Only use points where we have valid data after shifting
        valid_mask = (time_shifted >= time_s[0]) & (time_shifted <= time_s[-1])
        
        compensated_signal = np.zeros_like(fir_output)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            compensated_signal[valid_indices] = np.interp(
                time_shifted[valid_indices], 
                time_s, 
                fir_output
            )
        
        print(f"   ✅ Phase Compensation Applied")
        
        return {
            **signals,  # Keep all previous signals
            'phase_compensated': compensated_signal,
            'phase_shift_deg': phase_shift_deg,
            'delay_ms': delay_ms,
            'delay_seconds': delay_seconds
        }
    
    def step4_zero_crossing_detection(self, signals: Dict) -> Dict:
        """
        Step 4: Zero crossing detection on phase-compensated signal
        """
        print(f"\n🎯 Step 4: Zero Crossing Detection")
        
        time_s = signals['time_s']
        compensated_signal = signals['phase_compensated']
        
        # Process signal for zero crossings (simulate 400µs task cycles)
        task_cycle_time_ns = 400000  # 400µs in nanoseconds
        current_time_ns = task_cycle_time_ns  # Start after first cycle
        
        zero_crossings = []
        frequency_measurements = []
        
        # Sample every 400µs for zero crossing detection
        sample_interval_s = 400e-6  # 400µs
        sample_times = np.arange(0, time_s[-1], sample_interval_s)
        
        for sample_time in sample_times:
            # Get voltage at this time point
            voltage = np.interp(sample_time, time_s, compensated_signal)
            sample_time_ns = int(sample_time * 1e9)
            
            # Call zero crossing detector
            try:
                detected, last_crossing_ns, voltage_sign, avg_time_ns = self.zero_crossing.Call(
                    voltage, sample_time_ns)
                
                if detected and avg_time_ns > 0:
                    crossing_time_s = last_crossing_ns / 1e9
                    
                    # Calculate frequency (TwinCAT formula)
                    frequency_hz = 500000000.0 / avg_time_ns if avg_time_ns > 1000000 else 0.0
                    
                    if 10.0 <= frequency_hz <= 100.0:  # Valid frequency range
                        zero_crossings.append({
                            'time_s': crossing_time_s,
                            'voltage_sign': voltage_sign,
                            'frequency_hz': frequency_hz
                        })
                        frequency_measurements.append(frequency_hz)
                        
            except Exception as e:
                continue
        
        # Calculate timing accuracy vs theoretical 50Hz crossings
        timing_analysis = self._calculate_timing_accuracy(zero_crossings)
        
        print(f"   ✅ Zero Crossings Detected: {len(zero_crossings)}")
        print(f"   ✅ Frequency Measurements: {len(frequency_measurements)}")
        if timing_analysis['measurement_count'] > 0:
            print(f"   🎯 Timing Accuracy: ±{timing_analysis['avg_error_us']:.1f}µs avg, {timing_analysis['max_error_us']:.1f}µs max")
        
        return {
            **signals,  # Keep all previous signals
            'zero_crossings': zero_crossings,
            'frequency_measurements': frequency_measurements,
            'timing_accuracy': timing_analysis
        }
    
    def _calculate_timing_accuracy(self, zero_crossings: List[Dict]) -> Dict:
        """
        Calculate timing accuracy vs theoretical 50Hz zero crossings
        """
        if len(zero_crossings) == 0:
            return {
                'measurement_count': 0,
                'avg_error_us': 0.0,
                'max_error_us': 0.0,
                'rms_error_us': 0.0,
                'errors_us': []
            }
        
        # Generate theoretical 50Hz zero crossings
        # Start from first detected crossing and generate theoretical times
        first_detected_time = zero_crossings[0]['time_s']
        
        # Find the theoretical crossing closest to first detected
        # 50Hz has zero crossings every 10ms (half period)
        # Start from a reasonable base time
        base_theoretical_time = 0.0
        while base_theoretical_time < first_detected_time:
            base_theoretical_time += 0.010  # 10ms increments
        
        # If we overshot, go back one
        if base_theoretical_time - first_detected_time > 0.005:  # More than 5ms off
            base_theoretical_time -= 0.010
        
        # Calculate errors for each detected crossing
        timing_errors_us = []
        
        for crossing in zero_crossings:
            detected_time = crossing['time_s']
            
            # Find nearest theoretical crossing time
            # Theoretical crossings are at: base_time + n*0.010s
            crossing_number = round((detected_time - base_theoretical_time) / 0.010)
            nearest_theoretical_time = base_theoretical_time + (crossing_number * 0.010)
            
            # Calculate error in microseconds
            error_us = (detected_time - nearest_theoretical_time) * 1e6
            timing_errors_us.append(error_us)
        
        # Calculate statistics
        if len(timing_errors_us) > 0:
            avg_error_us = abs(np.mean(timing_errors_us))
            max_error_us = np.max(np.abs(timing_errors_us))
            rms_error_us = np.sqrt(np.mean(np.array(timing_errors_us)**2))
        else:
            avg_error_us = max_error_us = rms_error_us = 0.0
        
        return {
            'measurement_count': len(timing_errors_us),
            'avg_error_us': avg_error_us,
            'max_error_us': max_error_us,
            'rms_error_us': rms_error_us,
            'errors_us': timing_errors_us,
            'base_theoretical_time': base_theoretical_time
        }
    
    def step5_frequency_analysis(self, signals: Dict) -> Dict:
        """
        Step 5: Final frequency measurement analysis
        """
        print(f"\n📊 Step 5: Final Frequency Analysis")
        
        freq_measurements = signals['frequency_measurements']
        
        if len(freq_measurements) > 0:
            avg_frequency = np.mean(freq_measurements)
            std_frequency = np.std(freq_measurements)
            error_ppm = ((avg_frequency - 50.0) / 50.0) * 1000000
            
            print(f"   📈 Average Frequency: {avg_frequency:.4f}Hz")
            print(f"   📊 Std Deviation: {std_frequency:.4f}Hz") 
            print(f"   🎯 Error: {error_ppm:+.1f}ppm from 50Hz")
            
            analysis = {
                'avg_frequency': avg_frequency,
                'std_frequency': std_frequency,
                'error_ppm': error_ppm,
                'min_frequency': np.min(freq_measurements),
                'max_frequency': np.max(freq_measurements),
                'measurement_count': len(freq_measurements)
            }
        else:
            print(f"   ⚠️  No valid frequency measurements")
            analysis = {
                'avg_frequency': 0.0,
                'std_frequency': 0.0,
                'error_ppm': 0.0,
                'min_frequency': 0.0,
                'max_frequency': 0.0,
                'measurement_count': 0
            }
        
        return {
            **signals,  # Keep all previous data
            'frequency_analysis': analysis
        }
    
    def create_complete_visualization(self, data: Dict) -> go.Figure:
        """
        Create complete 5-step workflow visualization
        """
        # Create subplots for all 5 steps
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=[
                '📊 Step 1: Perfect 50Hz vs Your Generated Signal',
                '🔧 Step 2: FIR Filter Applied (Input vs Filtered Output)',
                '⚡ Step 3: Phase Shift Compensation (Delayed vs Compensated)',
                '🎯 Step 4: Zero Crossing Detection (Compensated Signal)',
                '📈 Step 5: Frequency Measurement Results'
            ],
            vertical_spacing=0.06,
            row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        time_ms = data['time_s'] * 1000  # Convert to milliseconds
        
        # Step 1: Original signals comparison
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['perfect_50hz'],
            name='Perfect 50Hz', line=dict(color='blue', width=2),
            showlegend=True
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['your_signal'],
            name='Your Signal (Einbrüche + Harmonics)', 
            line=dict(color='red', width=1.5, dash='dot'),
            showlegend=True
        ), row=1, col=1)
        
        # Step 2: FIR filter comparison
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['your_signal'],
            name='Input Signal', line=dict(color='red', width=1.5),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['fir_output'],
            name='FIR Filtered Output', line=dict(color='green', width=2.5),
            showlegend=True
        ), row=2, col=1)
        
        # Step 3: Phase compensation
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['fir_output'],
            name='FIR Output (Delayed)', line=dict(color='green', width=2, dash='dash'),
            showlegend=False
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['phase_compensated'],
            name=f'Phase Compensated (+{data["delay_ms"]:.1f}ms)', 
            line=dict(color='orange', width=2.5),
            showlegend=True
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['perfect_50hz'],
            name='Perfect 50Hz Reference', line=dict(color='blue', width=1, dash='dot'),
            opacity=0.7, showlegend=False
        ), row=3, col=1)
        
        # Step 4: Zero crossings
        fig.add_trace(go.Scatter(
            x=time_ms, y=data['phase_compensated'],
            name='Phase Compensated Signal', line=dict(color='orange', width=2),
            showlegend=False
        ), row=4, col=1)
        
        # Mark zero crossings
        if len(data['zero_crossings']) > 0:
            zc_times_ms = [zc['time_s'] * 1000 for zc in data['zero_crossings']]
            zc_colors = ['purple' if zc['voltage_sign'] else 'brown' for zc in data['zero_crossings']]
            
            fig.add_trace(go.Scatter(
                x=zc_times_ms, y=[0] * len(zc_times_ms),
                mode='markers', name='Zero Crossings',
                marker=dict(color=zc_colors, size=10, symbol='diamond'),
                showlegend=True
            ), row=4, col=1)
            
            # Add theoretical 50Hz reference crossings
            if 'timing_accuracy' in data and data['timing_accuracy']['measurement_count'] > 0:
                base_time_s = data['timing_accuracy']['base_theoretical_time']
                base_time_ms = base_time_s * 1000
            else:
                base_time_ms = zc_times_ms[0] if len(zc_times_ms) > 0 else 0
            
            # Add theoretical crossing reference lines every 10ms
            for i in range(-2, int(data['duration_ms'] / 10) + 3):  # Extra range for visibility
                theoretical_time_ms = base_time_ms + (i * 10.0)
                if 0 <= theoretical_time_ms <= data['duration_ms']:
                    fig.add_vline(
                        x=theoretical_time_ms, line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.4, row=4, col=1
                    )

            # Add timing accuracy annotation
            if 'timing_accuracy' in data and data['timing_accuracy']['measurement_count'] > 0:
                accuracy = data['timing_accuracy']
                fig.add_annotation(
                    x=0.98, y=0.42, xref='paper', yref='paper',
                    text=f"⏱️ Timing Accuracy vs 50Hz:<br>" +
                         f"Avg Error: ±{accuracy['avg_error_us']:.1f}µs<br>" +
                         f"Max Error: {accuracy['max_error_us']:.1f}µs<br>" +
                         f"RMS Error: {accuracy['rms_error_us']:.1f}µs<br>" +
                         f"Measurements: {accuracy['measurement_count']}",
                    showarrow=False, bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='purple', borderwidth=2, font=dict(size=11),
                    xanchor='right', align='left'
                )
        
        # Step 5: Frequency measurements
        if len(data['frequency_measurements']) > 0:
            freq_times = [zc['time_s'] * 1000 for zc in data['zero_crossings']]
            frequencies = data['frequency_measurements']
            
            fig.add_trace(go.Scatter(
                x=freq_times, y=frequencies,
                mode='lines+markers', name='Measured Frequency',
                line=dict(color='darkblue', width=3),
                marker=dict(size=6), showlegend=True
            ), row=5, col=1)
            
            # Add 50Hz reference line
            fig.add_hline(
                y=50.0, line=dict(color='gray', width=2, dash='dash'),
                row=5, col=1
            )
            
            # Add frequency statistics
            analysis = data['frequency_analysis']
            fig.add_annotation(
                x=0.02, y=0.02, xref='paper', yref='paper',
                text=f"Avg: {analysis['avg_frequency']:.4f}Hz<br>" +
                     f"Error: {analysis['error_ppm']:+.1f}ppm<br>" +
                     f"Count: {analysis['measurement_count']}",
                showarrow=False, bgcolor='rgba(255,255,255,0.9)',
                bordercolor='darkblue', borderwidth=1, font=dict(size=10)
            )
        
        # Add zero reference lines to all voltage plots (rows 1-4)
        for row in range(1, 5):
            fig.add_hline(
                y=0, line=dict(color='black', width=1, dash='solid'),
                opacity=0.3, row=row, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="TwinCAT FIR Filter Complete Workflow: 5-Step Processing Chain<br>" +
                     f"<sub>136th Order FIR → Phase Compensation ({data['delay_ms']:.1f}ms) → Zero Crossing → Frequency</sub>",
                x=0.5, font=dict(size=16)
            ),
            height=1200,
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor='left'),
            plot_bgcolor='white'
        )
        
        # Update axes
        for row in range(1, 6):
            fig.update_xaxes(
                title_text="Time [ms]" if row == 5 else "",
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                row=row, col=1
            )
            
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=3, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=4, col=1)
        fig.update_yaxes(title_text="Frequency [Hz]", row=5, col=1)
        
        return fig
    
    def run_complete_workflow(self, duration_ms: float = 100.0, save_html: bool = True) -> Dict:
        """
        Run the complete 5-step TwinCAT FIR workflow
        """
        print(f"🎯 TwinCAT FIR Filter Complete Workflow Analysis")
        print(f"=" * 60)
        
        # Execute all 5 steps
        data = self.step1_generate_signals(duration_ms)
        data = self.step2_apply_fir_filter(data)
        data = self.step3_phase_compensation(data)
        data = self.step4_zero_crossing_detection(data)
        data = self.step5_frequency_analysis(data)
        
        # Create visualization
        print(f"\n📈 Creating Complete Workflow Visualization...")
        fig = self.create_complete_visualization(data)
        
        # Save HTML
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twincat_fir_complete_workflow_{timestamp}.html"
            fig.write_html(filename)
            print(f"   💾 Saved: {filename}")
        
        # Print final summary
        print(f"\n✅ TwinCAT FIR Workflow Complete!")
        print(f"   Duration: {duration_ms}ms")
        if 'frequency_analysis' in data and data['frequency_analysis']['measurement_count'] > 0:
            analysis = data['frequency_analysis']
            print(f"   Frequency: {analysis['avg_frequency']:.4f}Hz (Error: {analysis['error_ppm']:+.1f}ppm)")
            print(f"   Phase Compensation: {data['delay_ms']:.2f}ms @ 50Hz")
            print(f"   Zero Crossings: {len(data['zero_crossings'])}")
            
            if 'timing_accuracy' in data:
                accuracy = data['timing_accuracy']
                print(f"   Timing Accuracy: ±{accuracy['avg_error_us']:.1f}µs avg, {accuracy['max_error_us']:.1f}µs max")
            
            # Validation
            freq_ok = abs(analysis['error_ppm']) < 200  # ±200ppm tolerance
            timing_ok = 'timing_accuracy' in data and data['timing_accuracy']['avg_error_us'] < 100  # <100µs
            print(f"   Validation: {'✅ PASS' if freq_ok and timing_ok else '❌ FAIL'}")
        
        return {
            'data': data,
            'figure': fig,
            'summary': {
                'duration_ms': duration_ms,
                'fir_configured': self.fir_filter.bConfigured,
                'phase_compensation_ms': data.get('delay_ms', 0),
                'frequency_analysis': data.get('frequency_analysis', {}),
                'timing_accuracy': data.get('timing_accuracy', {}),
                'zero_crossings_count': len(data.get('zero_crossings', []))
            }
        }


def main():
    """
    Main function to run the complete TwinCAT FIR workflow visualization
    """
    print("🎯 TwinCAT FIR Filter Complete Workflow Visualization")
    print("=" * 65)
    
    # Create workflow visualizer
    viz = TwinCATFIRWorkflowViz()
    
    # Run complete workflow
    results = viz.run_complete_workflow(duration_ms=100.0, save_html=True)
    
    print(f"\n🎯 All 5 steps completed and visualized!")
    print(f"   Check the HTML file for interactive plots")
    print(f"   Each step shows the exact TwinCAT processing")
    print(f"   Timing accuracy shows how close to perfect 50Hz!")


if __name__ == "__main__":
    main()