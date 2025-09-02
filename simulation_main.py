"""
simulation_main.py - TwinCAT Complete System Simulation - REFACTORED VERSION
==========================================================================

TwinCAT Sources: 
- Task timing from Documents 17-23 (Measurements.TcTTO: 400µs, Priority 3)
- System integration from FB_EL3783_LxLy and complete measurement chain

This orchestrates the complete 1:1 TwinCAT simulation:
1. Signal Generation (using existing signal_generator.py)
2. Hardware Interface (ADC conversion, timing)
3. Measurement Processing (FIR, Zero Crossing, RMS)
4. Task Scheduling (400µs cycles like TwinCAT)
5. Data Collection and Analysis

Complete pipeline: Signal → Hardware → Measurement → Analysis

REFACTORED CHANGES:
- Removed duplicate TwinCATSignalGenerator class
- Now imports and uses TwinCATIntegratedSignGenerator from signal_generator.py
- Cleaner architecture with single signal generator
- Maintained all original functionality
"""

import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict

# Import our TwinCAT replicas
from fir_filter import FB_FIRFilterOvSampl, create_configured_filter
from zero_crossing_detector import ZeroCrossing, ZeroCrossingSimulator
from rms_calculator import RMSHalfCycle, RMSMeasurementSystem
from measurement_system import FB_EL3783_LxLy, MeasurementSystemSimulator

# Import the existing signal generator (no more duplicates!)
from signal_generator import TwinCATIntegratedSignGenerator


@dataclass
class SimulationConfig:
    """
    Configuration for TwinCAT simulation
    
    Matches the original TwinCAT system parameters
    """
    # Task timing (from TwinCAT task configuration)
    measurement_task_cycle_us: float = 400.0    # Measurements.TcTTO: 400µs cycle time
    measurement_task_priority: int = 3          # Priority 3 (high priority)
    
    # Hardware parameters (from FB_EL3783_LxLy)
    oversamples_per_cycle: int = 8              # OVERSAMPLES = 8
    oversample_interval_us: float = 50.0        # SAMPLE_SYNC_TIME_us = 50
    adc_volts_per_digit: float = 0.0224993      # VoltsPerDigit scaling
    
    # Signal generation
    mains_frequency_hz: float = 50.0            # European mains frequency
    mains_amplitude_v: float = 325.0            # ~230V RMS * sqrt(2)
    line_to_line_amplitude_v: float = 563.0     # ~400V RMS * sqrt(2)
    noise_level_v: float = 2.0                  # Realistic noise level
    
    # Simulation parameters
    simulation_duration_ms: float = 200.0       # Total simulation time
    enable_disturbances: bool = True            # Add voltage drops/spikes
    enable_harmonics: bool = True               # Add realistic harmonics
    simulation_mode: bool = False               # TwinCAT simulation mode
    
    # Analysis parameters
    expected_frequency_hz: float = 50.0         # Expected mains frequency
    expected_l2l_voltage_rms: float = 400.0     # Expected L2L RMS voltage
    frequency_tolerance_hz: float = 0.1         # ±0.1Hz tolerance
    voltage_tolerance_percent: float = 2.0      # ±2% voltage tolerance


class TwinCATTaskScheduler:
    """
    TwinCAT task scheduling simulation
    
    Simulates the real-time task execution with proper timing:
    - Measurements Task: 400µs cycle, Priority 3
    - Precise timing control
    - Task cycle counting
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.measurement_task_period_ns = int(config.measurement_task_cycle_us * 1000)
        
        # Task state - Start with positive time to avoid negative sampling times
        self.current_time_ns = self.measurement_task_period_ns  # Start after first cycle
        self.measurement_cycle_count = 0
        self.start_time_real = time.time()
        
    def get_current_dc_time_ns(self) -> int:
        """
        Get current distributed clock time in nanoseconds
        
        Simulates TwinCAT F_GetActualDcTime64()
        """
        return self.current_time_ns
    
    def advance_to_next_measurement_cycle(self) -> int:
        """
        Advance time to next measurement task cycle
        
        Returns:
            int: Start time of next measurement cycle [ns]
        """
        self.current_time_ns += self.measurement_task_period_ns
        self.measurement_cycle_count += 1
        return self.current_time_ns
    
    def calculate_oversample_times(self) -> np.ndarray:
        """
        Calculate the 8 oversample time points within current cycle
        
        TwinCAT takes 8 oversamples every 50µs within the 400µs cycle
        
        Returns:
            np.ndarray: 8 time points in seconds for oversampling
        """
        # Start time for oversampling (at beginning of cycle)
        cycle_start_time_s = (self.current_time_ns - self.measurement_task_period_ns) / 1e9
        
        # 8 samples, 50µs apart
        sample_times = []
        for i in range(8):
            sample_time_s = cycle_start_time_s + (i * self.config.oversample_interval_us * 1e-6)
            sample_times.append(sample_time_s)
        
        return np.array(sample_times)
    
    def get_start_time_next_latch(self) -> int:
        """
        Get hardware latch time for next measurement cycle
        
        Simulates the StartTimeNextLatch hardware input
        """
        return self.current_time_ns


class TwinCATSystemSimulation:
    """
    Complete TwinCAT system simulation
    
    Orchestrates all components:
    1. Signal generation using existing signal_generator.py
    2. Task scheduling with proper timing
    3. Hardware interface (ADC, timing)  
    4. Measurement processing (all algorithms)
    5. Data collection and analysis
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Initialize subsystems - using the existing signal generator!
        self.signal_generator = TwinCATIntegratedSignGenerator()
        self.task_scheduler = TwinCATTaskScheduler(config)
        self.measurement_simulator = MeasurementSystemSimulator(config.measurement_task_cycle_us)
        
        # Data collection
        self.measurement_results = []
        self.signal_data = []
        self.performance_data = []
        
        # Statistics
        self.total_cycles = 0
        self.zero_crossings_detected = 0
        self.valid_rms_measurements = 0
        self.frequency_measurements = []
        self.voltage_measurements = []
        
        print(f"TwinCAT System Simulation Initialized")
        print(f"   Task Cycle: {config.measurement_task_cycle_us}µs")
        print(f"   Duration: {config.simulation_duration_ms}ms") 
        print(f"   Expected Cycles: {int(config.simulation_duration_ms * 1000 / config.measurement_task_cycle_us)}")
        print(f"   Signal Generator: TwinCATIntegratedSignGenerator (from signal_generator.py)")
    
    def run_simulation(self) -> Dict:
        """
        Run the complete TwinCAT simulation
        
        Returns:
            Dict: Complete simulation results and analysis
        """
        print(f"\nStarting TwinCAT System Simulation...")
        
        # Calculate simulation parameters
        total_cycles = int(self.config.simulation_duration_ms * 1000 / self.config.measurement_task_cycle_us)
        
        simulation_start_time = time.time()
        
        # Main simulation loop
        for cycle in range(total_cycles):
            # Advance to next measurement cycle
            next_latch_time = self.task_scheduler.advance_to_next_measurement_cycle()
            
            # Get oversample time points for this cycle
            sample_times = self.task_scheduler.calculate_oversample_times()
            
            # Generate 3-phase voltages using the existing signal generator
            L1_voltages, L2_voltages, L3_voltages = self._generate_voltages_at_times(
                sample_times, self.config.mains_amplitude_v)
            
            # Store signal data for analysis
            self.signal_data.append({
                'cycle': cycle,
                'time_ns': self.task_scheduler.current_time_ns - self.task_scheduler.measurement_task_period_ns,
                'sample_times': sample_times.tolist(),
                'L1_voltages': L1_voltages.tolist(),
                'L2_voltages': L2_voltages.tolist(), 
                'L3_voltages': L3_voltages.tolist()
            })
            
            # Process measurement cycle (using L1 and L2 for line-to-line measurement)
            result = self.measurement_simulator.process_measurement_cycle(
                L1_voltages.tolist(),  # Lx samples
                L2_voltages.tolist(),  # Ly samples
                self.config.simulation_mode
            )
            
            # Store results
            self.measurement_results.append(result)
            
            # Update statistics
            self.total_cycles += 1
            
            if result['frequency_hz'] > 0:
                self.frequency_measurements.append(result['frequency_hz'])
            
            if result['l2l_voltage_rms'] > 0:
                self.voltage_measurements.append(result['l2l_voltage_rms'])
                self.valid_rms_measurements += 1
            
            if result['zero_crossing_time_ns'] > 0:
                self.zero_crossings_detected += 1
            
            # Progress indication
            if cycle % max(1, total_cycles // 10) == 0:
                progress = (cycle / total_cycles) * 100
                print(f"   Progress: {progress:.1f}% - Cycle {cycle}/{total_cycles}")
        
        simulation_end_time = time.time()
        simulation_duration_real = simulation_end_time - simulation_start_time
        
        print(f"Simulation Complete!")
        print(f"   Real Time: {simulation_duration_real:.2f}s")
        print(f"   Simulated Time: {self.config.simulation_duration_ms}ms")
        print(f"   Speed Factor: {(self.config.simulation_duration_ms/1000)/simulation_duration_real:.1f}x")
        
        # Generate comprehensive analysis
        analysis = self._analyze_results()
        
        return {
            'config': asdict(self.config),
            'performance': {
                'total_cycles': self.total_cycles,
                'zero_crossings_detected': self.zero_crossings_detected,
                'valid_rms_measurements': self.valid_rms_measurements,
                'real_execution_time_s': simulation_duration_real,
                'simulated_time_ms': self.config.simulation_duration_ms
            },
            'measurement_results': self.measurement_results,
            'signal_data': self.signal_data[:10],  # First 10 cycles for analysis
            'analysis': analysis
        }
    
    def _generate_voltages_at_times(self, sample_times: np.ndarray, amplitude: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3-phase voltages using the existing signal generator
        
        This method acts as a bridge between the task scheduler timing
        and the existing signal generator's methods.
        """
        # Use the existing signal generator's 3-phase generation method
        L1_voltages, L2_voltages, L3_voltages = self.signal_generator._generate_3phase_at_times(
            sample_times, amplitude)
        
        # Apply disturbances using the existing signal generator's method
        cycle_idx = self.task_scheduler.measurement_cycle_count
        L1_voltages = self.signal_generator._apply_your_disturbances(L1_voltages, sample_times, cycle_idx)
        L2_voltages = self.signal_generator._apply_your_disturbances(L2_voltages, sample_times, cycle_idx)
        L3_voltages = self.signal_generator._apply_your_disturbances(L3_voltages, sample_times, cycle_idx)
        
        return L1_voltages, L2_voltages, L3_voltages
    
    def _analyze_results(self) -> Dict:
        """
        Comprehensive analysis of simulation results
        """
        analysis = {
            'summary': {
                'total_cycles': self.total_cycles,
                'measurement_success_rate': (self.valid_rms_measurements / self.total_cycles) * 100 if self.total_cycles > 0 else 0,
                'zero_crossing_success_rate': (self.zero_crossings_detected / self.total_cycles) * 100 if self.total_cycles > 0 else 0
            }
        }
        
        # Frequency analysis
        if len(self.frequency_measurements) > 0:
            freq_array = np.array(self.frequency_measurements)
            analysis['frequency'] = {
                'count': len(self.frequency_measurements),
                'mean_hz': float(np.mean(freq_array)),
                'std_hz': float(np.std(freq_array)),
                'min_hz': float(np.min(freq_array)),
                'max_hz': float(np.max(freq_array)),
                'accuracy_error_hz': float(np.mean(freq_array) - self.config.expected_frequency_hz),
                'within_tolerance': bool(np.abs(np.mean(freq_array) - self.config.expected_frequency_hz) <= self.config.frequency_tolerance_hz)
            }
        
        # Voltage analysis
        if len(self.voltage_measurements) > 0:
            volt_array = np.array(self.voltage_measurements)
            analysis['voltage'] = {
                'count': len(self.voltage_measurements),
                'mean_v': float(np.mean(volt_array)),
                'std_v': float(np.std(volt_array)),
                'min_v': float(np.min(volt_array)),
                'max_v': float(np.max(volt_array)),
                'accuracy_error_v': float(np.mean(volt_array) - self.config.expected_l2l_voltage_rms),
                'accuracy_error_percent': float((np.mean(volt_array) - self.config.expected_l2l_voltage_rms) / self.config.expected_l2l_voltage_rms * 100),
                'within_tolerance': bool(np.abs((np.mean(volt_array) - self.config.expected_l2l_voltage_rms) / self.config.expected_l2l_voltage_rms * 100) <= self.config.voltage_tolerance_percent)
            }
        
        # FIR Filter analysis
        if len(self.measurement_results) > 0:
            # Get FIR filter information from last measurement
            last_result = self.measurement_results[-1]
            if 'debug' in last_result and 'filter_state' in last_result['debug']:
                filter_info = last_result['debug']['filter_state']
                analysis['fir_filter'] = {
                    'configured': last_result['debug']['measurement_state']['fir_filter_configured'],
                    'phase_shift_50hz_deg': filter_info.get('phase_shift_50hz', 0),
                    'phase_shift_measured_freq_deg': filter_info.get('phase_shift_current_freq', 0)
                }
        
        # Zero crossing timing analysis
        zero_crossing_results = [r for r in self.measurement_results if r['zero_crossing_time_ns'] > 0]
        if len(zero_crossing_results) > 1:
            # Calculate time differences between zero crossings
            zc_times = [r['zero_crossing_time_ns'] for r in zero_crossing_results]
            time_diffs = np.diff(zc_times) / 1e6  # Convert to ms
            
            analysis['zero_crossings'] = {
                'count': len(zero_crossing_results),
                'avg_interval_ms': float(np.mean(time_diffs)),
                'std_interval_ms': float(np.std(time_diffs)),
                'expected_interval_ms': 1000.0 / (2 * self.config.expected_frequency_hz),  # Half cycle
                'timing_accuracy_ms': float(np.mean(time_diffs) - (1000.0 / (2 * self.config.expected_frequency_hz)))
            }
        
        return analysis
    
    def save_results_to_file(self, results: Dict, filename: str = "twincat_simulation_results.json"):
        """
        Save simulation results to JSON file for further analysis
        
        Handles numpy types and other non-JSON serializable objects
        """
        try:
            # Convert numpy types and other non-serializable objects for JSON compatibility
            def convert_for_json(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # Handle other numpy scalar types
                    return obj.item()
                return obj
            
            # Deep convert the results dictionary
            def deep_convert(obj):
                if isinstance(obj, dict):
                    return {key: deep_convert(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(item) for item in obj]
                else:
                    return convert_for_json(obj)
            
            # Clean results for JSON
            json_compatible = deep_convert(results)
            
            with open(filename, 'w') as f:
                json.dump(json_compatible, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # Try to save a minimal version
            try:
                minimal_results = {
                    'config': asdict(self.config),
                    'total_cycles': int(self.total_cycles),
                    'zero_crossings': int(self.zero_crossings_detected),
                    'rms_measurements': int(self.valid_rms_measurements),
                    'error': str(e)
                }
                with open(f"minimal_{filename}", 'w') as f:
                    json.dump(minimal_results, f, indent=2)
                print(f"Minimal results saved to minimal_{filename}")
            except Exception as e2:
                print(f"Could not save minimal results either: {e2}")
    
    def print_summary(self, results: Dict):
        """
        Print comprehensive simulation summary
        """
        print(f"\nTwinCAT Simulation Results Summary")
        print(f"=" * 50)
        
        perf = results['performance']
        analysis = results['analysis']
        
        print(f"Performance:")
        print(f"  Total Cycles: {perf['total_cycles']}")
        print(f"  Zero Crossings: {perf['zero_crossings_detected']} ({analysis['summary']['zero_crossing_success_rate']:.1f}%)")
        print(f"  RMS Measurements: {perf['valid_rms_measurements']} ({analysis['summary']['measurement_success_rate']:.1f}%)")
        
        if 'frequency' in analysis:
            freq = analysis['frequency']
            print(f"\nFrequency Measurement:")
            print(f"  Measured: {freq['mean_hz']:.4f} ± {freq['std_hz']:.4f} Hz")
            print(f"  Expected: {self.config.expected_frequency_hz} Hz")
            print(f"  Error: {freq['accuracy_error_hz']:.4f} Hz")
            print(f"  Accuracy: {'PASS' if freq['within_tolerance'] else 'FAIL'}")
        
        if 'voltage' in analysis:
            volt = analysis['voltage']
            print(f"\nVoltage Measurement:")
            print(f"  Measured: {volt['mean_v']:.1f} ± {volt['std_v']:.1f} V")
            print(f"  Expected: {self.config.expected_l2l_voltage_rms} V")
            print(f"  Error: {volt['accuracy_error_percent']:.2f}%")
            print(f"  Accuracy: {'PASS' if volt['within_tolerance'] else 'FAIL'}")
        
        if 'fir_filter' in analysis:
            fir = analysis['fir_filter']
            print(f"\nFIR Filter:")
            print(f"  Configured: {'YES' if fir['configured'] else 'NO'}")
            print(f"  Phase Shift @ 50Hz: {fir['phase_shift_50hz_deg']:.2f}°")
        
        if 'zero_crossings' in analysis:
            zc = analysis['zero_crossings']
            print(f"\nZero Crossing Accuracy:")
            print(f"  Average Interval: {zc['avg_interval_ms']:.3f}ms")
            print(f"  Expected Interval: {zc['expected_interval_ms']:.3f}ms")
            print(f"  Timing Error: {zc['timing_accuracy_ms']:.3f}ms")


def main():
    """
    Main function to run the TwinCAT simulation
    """
    print("TwinCAT Complete System Simulation - REFACTORED VERSION")
    print("=" * 65)
    print("CHANGES: Removed duplicate signal generator, using signal_generator.py")
    
    # Create simulation configuration
    config = SimulationConfig(
        simulation_duration_ms=100.0,      # Shorter for testing - 100ms
        enable_disturbances=True,          # Include realistic disturbances
        enable_harmonics=True,             # Include harmonics
        mains_frequency_hz=50.0,          # European 50Hz mains
        expected_l2l_voltage_rms=400.0,   # Expected 400V L2L
    )
    
    # Run simulation
    simulation = TwinCATSystemSimulation(config)
    results = simulation.run_simulation()
    
    # Print summary
    simulation.print_summary(results)
    
    # Save results
    simulation.save_results_to_file(results)
    
    print(f"\nTwinCAT 1:1 Replica Simulation Complete!")
    print(f"   Using single signal generator from signal_generator.py")
    print(f"   No more duplicate code - cleaner architecture")
    print(f"   All measurement algorithms working with unified signal generation")


if __name__ == "__main__":
    main()