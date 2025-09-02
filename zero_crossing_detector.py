"""
zero_crossing_detector.py - TwinCAT ZeroCrossing 1:1 Replica
=============================================================

Original TwinCAT Source: ZeroCrossing.TcPOU (Document 15)
Company: Schlatter Industries AG
Application: SWEP 30 Welding System

This is an exact 1:1 replica of the TwinCAT zero crossing detection
used for precise mains frequency measurement and welding synchronization.
Features sub-sample accuracy through linear interpolation.
"""

import numpy as np
from typing import Tuple


class ZeroCrossing:
    """
    1:1 Replica of TwinCAT ZeroCrossing Function Block
    
    Original TwinCAT Purpose:
    - Calculate time of last zero crossing
    - Determine polarity of last half wave  
    - Average time between two zero crossings
    - RMS value of input voltage over last half wave
    
    Key Features:
    - Sub-sample accuracy through linear interpolation
    - 10-point averaging for stable frequency measurement
    - 400µs task cycle time precision
    """
    
    # TwinCAT Constants - exact 1:1
    TASK_CYCLETIME_ns = 400000.0    # Original: TASK_CYCLETIME_ns : LREAL := 400000 (400µs)
    nrOfSamples = 10                # Original: nrOfSamples : INT := 10
    
    def __init__(self):
        """
        Initialize ZeroCrossing instance
        
        TwinCAT VAR section replica:
        - voltageLastCycle: LREAL - Voltage sample value from last task cycle
        - bVoltageSignLastCycle: BOOL - Polarity of last half wave from last cycle
        - lastZeroCrossingLastCycle: ULINT - Time of last zero crossing from last cycle  
        - timeBetweenZeroCrossing: ARRAY[0..9] OF ULINT - Times between crossings for averaging
        - sampleIdx: INT - Index pointing to next array element for measurement
        """
        # Voltage sample value from last task cycle
        self.voltageLastCycle = 0.0
        
        # Polarity of last half wave (TRUE if positive) from last task cycle
        self.bVoltageSignLastCycle = False
        
        # Time of last zero crossing from last task cycle
        self.lastZeroCrossingLastCycle = 0
        
        # Array of times between two zero crossings [ns] for averaging
        self.timeBetweenZeroCrossing = np.zeros(self.nrOfSamples, dtype=np.uint64)
        
        # Index pointing to array element for next measurement value
        self.sampleIdx = 0
    
    def Call(self, voltage: float, sampleTime: int) -> Tuple[bool, int, bool, int]:
        """
        Zero crossing detection function call
        
        TwinCAT Method: Call : BOOL
        
        Args:
            voltage: Current voltage sample value
            sampleTime: Time when voltage sample was captured [ns]
            
        Returns:
            Tuple containing:
            - bool: TRUE if zero crossing detected, FALSE otherwise
            - int: Time of last zero crossing [ns]
            - bool: Polarity of last half wave (TRUE if positive)
            - int: Average time between two zero crossings [ns] (over 10 measurements)
            
        Original TwinCAT VAR_IN_OUT:
        - lastZeroCrossing: Time of last zero crossing
        - voltageSign: Polarity of last half wave
        - averageTimeBetweenZeroCrossing: Average time between crossings
        
        TwinCAT Logic:
        1. Determine current polarity: bVoltageSign := voltage > 0
        2. Check for zero crossing: bVoltageSign <> bVoltageSignLastCycle
        3. Linear interpolation for exact crossing time
        4. 10-point averaging for stable frequency measurement
        """
        # Determine current polarity - TwinCAT: bVoltageSign := voltage > 0
        bVoltageSign = voltage > 0
        
        # Initialize return values
        lastZeroCrossing = 0
        voltageSign = self.bVoltageSignLastCycle
        averageTimeBetweenZeroCrossing = 0
        
        # Check for zero crossing - TwinCAT: IF bVoltageSign <> bVoltageSignLastCycle THEN
        if bVoltageSign != self.bVoltageSignLastCycle:
            # Calculate ratio for linear interpolation
            # TwinCAT: IF (voltageLastCycle - voltage) <> 0 THEN
            if (self.voltageLastCycle - voltage) != 0:
                # TwinCAT: ratio := TO_REAL(voltageLastCycle / (voltageLastCycle - voltage))
                ratio = float(self.voltageLastCycle / (self.voltageLastCycle - voltage))
            else:
                # TwinCAT: ratio := 0
                ratio = 0.0
            
            # Calculate exact zero crossing time using linear interpolation
            # TwinCAT: lastZeroCrossing := sampleTime + TO_ULINT((ratio - 1) * TASK_CYCLETIME_ns)
            lastZeroCrossing = int(sampleTime + int((ratio - 1) * self.TASK_CYCLETIME_ns))
            
            # Store polarity of last voltage half wave
            # TwinCAT: voltageSign := bVoltageSignLastCycle
            voltageSign = self.bVoltageSignLastCycle
            
            # 10-point averaging for stable frequency measurement
            # TwinCAT: timeBetweenZeroCrossing[sampleIdx] := lastZeroCrossing - lastZeroCrossingLastCycle
            self.timeBetweenZeroCrossing[self.sampleIdx] = lastZeroCrossing - self.lastZeroCrossingLastCycle
            
            # Increment sample index with modulo
            # TwinCAT: sampleIdx := (sampleIdx + 1) MOD nrOfSamples
            self.sampleIdx = (self.sampleIdx + 1) % self.nrOfSamples
            
            # Calculate average over 10 half waves
            # TwinCAT: FOR idx := 0 TO (nrOfSamples - 1) DO sumForAvg := sumForAvg + timeBetweenZeroCrossing[idx]
            sumForAvg = 0
            for idx in range(self.nrOfSamples):
                sumForAvg += self.timeBetweenZeroCrossing[idx]
            
            # TwinCAT: averageTimeBetweenZeroCrossing := sumForAvg / TO_ULINT(nrOfSamples)
            averageTimeBetweenZeroCrossing = int(sumForAvg // self.nrOfSamples)
            
            # Store zero crossing time for next cycle
            # TwinCAT: lastZeroCrossingLastCycle := lastZeroCrossing
            self.lastZeroCrossingLastCycle = lastZeroCrossing
            
            # Return TRUE - zero crossing detected
            zero_crossing_detected = True
        else:
            # No zero crossing detected
            zero_crossing_detected = False
            # Keep previous values for return
            lastZeroCrossing = self.lastZeroCrossingLastCycle
            voltageSign = self.bVoltageSignLastCycle
            averageTimeBetweenZeroCrossing = int(np.sum(self.timeBetweenZeroCrossing) // self.nrOfSamples)
        
        # Store values for next cycle - TwinCAT: bVoltageSignLastCycle := bVoltageSign
        self.bVoltageSignLastCycle = bVoltageSign
        # TwinCAT: voltageLastCycle := voltage
        self.voltageLastCycle = voltage
        
        # Return tuple: (detected, lastZeroCrossing, voltageSign, averageTime)
        return (zero_crossing_detected, lastZeroCrossing, voltageSign, averageTimeBetweenZeroCrossing)
    
    def calculate_frequency_from_average_time(self, averageTimeBetweenZeroCrossing: int) -> float:
        """
        Calculate frequency from average time between zero crossings
        
        This matches the frequency calculation used in FB_EL3783_LxLy:
        _frequency := 500000000.0 / TO_LREAL(_averageTimeBetweenZeroCrossing)
        
        Args:
            averageTimeBetweenZeroCrossing: Average time between crossings [ns]
            
        Returns:
            float: Frequency in [Hz]
        """
        if averageTimeBetweenZeroCrossing > 0:
            # TwinCAT calculation: 500000000.0 / averageTime
            # 500MHz = 0.5 / 1ns, so frequency = 0.5 / (time_in_seconds)
            return 500000000.0 / float(averageTimeBetweenZeroCrossing)
        else:
            return 0.0
    
    def get_interpolation_details(self, voltage: float) -> dict:
        """
        Get detailed information about the interpolation calculation
        
        Useful for debugging and visualization of the sub-sample accuracy
        
        Args:
            voltage: Current voltage sample
            
        Returns:
            dict: Interpolation details including ratio and time offset
        """
        if (self.voltageLastCycle - voltage) != 0:
            ratio = float(self.voltageLastCycle / (self.voltageLastCycle - voltage))
            time_offset_ns = (ratio - 1) * self.TASK_CYCLETIME_ns
            
            return {
                'voltage_last': self.voltageLastCycle,
                'voltage_current': voltage,
                'ratio': ratio,
                'time_offset_ns': time_offset_ns,
                'interpolation_valid': True
            }
        else:
            return {
                'voltage_last': self.voltageLastCycle,
                'voltage_current': voltage, 
                'ratio': 0.0,
                'time_offset_ns': 0.0,
                'interpolation_valid': False
            }
    
    def reset(self):
        """
        Reset the zero crossing detector to initial state
        
        Useful for starting a new measurement sequence
        """
        self.voltageLastCycle = 0.0
        self.bVoltageSignLastCycle = False
        self.lastZeroCrossingLastCycle = 0
        self.timeBetweenZeroCrossing.fill(0)
        self.sampleIdx = 0


class ZeroCrossingSimulator:
    """
    Simulator for testing zero crossing detection with realistic timing
    
    Simulates the 400µs task cycle timing from TwinCAT measurement system
    """
    
    def __init__(self, task_cycle_time_us: float = 400.0):
        """
        Initialize zero crossing simulator
        
        Args:
            task_cycle_time_us: Task cycle time in microseconds (default: 400µs)
        """
        self.detector = ZeroCrossing()
        self.task_cycle_time_ns = int(task_cycle_time_us * 1000)  # Convert µs to ns
        self.current_time_ns = 0
        
        # Statistics
        self.zero_crossings_detected = 0
        self.frequency_measurements = []
    
    def process_voltage_sample(self, voltage: float) -> dict:
        """
        Process a voltage sample with realistic timing
        
        Args:
            voltage: Voltage sample value
            
        Returns:
            dict: Processing results including zero crossing detection
        """
        # Call zero crossing detector
        detected, last_crossing, voltage_sign, avg_time = self.detector.Call(voltage, self.current_time_ns)
        
        # Calculate frequency if zero crossing detected
        frequency = 0.0
        if detected and avg_time > 0:
            frequency = self.detector.calculate_frequency_from_average_time(avg_time)
            self.frequency_measurements.append(frequency)
            self.zero_crossings_detected += 1
        
        # Get interpolation details
        interpolation = self.detector.get_interpolation_details(voltage)
        
        # Advance time by one task cycle
        self.current_time_ns += self.task_cycle_time_ns
        
        return {
            'time_ns': self.current_time_ns - self.task_cycle_time_ns,  # Time of this sample
            'voltage': voltage,
            'zero_crossing_detected': detected,
            'last_zero_crossing_ns': last_crossing,
            'voltage_sign_positive': voltage_sign,
            'avg_time_between_crossings_ns': avg_time,
            'frequency_hz': frequency,
            'interpolation': interpolation,
            'total_crossings': self.zero_crossings_detected
        }
    
    def get_statistics(self) -> dict:
        """
        Get measurement statistics
        
        Returns:
            dict: Statistics about frequency measurements
        """
        if len(self.frequency_measurements) > 0:
            return {
                'total_zero_crossings': self.zero_crossings_detected,
                'frequency_measurements': len(self.frequency_measurements),
                'average_frequency': np.mean(self.frequency_measurements),
                'frequency_std': np.std(self.frequency_measurements),
                'min_frequency': np.min(self.frequency_measurements),
                'max_frequency': np.max(self.frequency_measurements)
            }
        else:
            return {
                'total_zero_crossings': self.zero_crossings_detected,
                'frequency_measurements': 0,
                'average_frequency': 0.0,
                'frequency_std': 0.0,
                'min_frequency': 0.0,
                'max_frequency': 0.0
            }


if __name__ == "__main__":
    """
    Test the zero crossing detector with simulated 50Hz signal
    """
    print("🎯 TwinCAT ZeroCrossing Detector Test")
    print("=" * 45)
    
    # Create simulator
    simulator = ZeroCrossingSimulator(task_cycle_time_us=400.0)  # 400µs like TwinCAT
    
    # Generate test signal: 50Hz sine wave with some noise
    duration_s = 0.1  # 100ms test
    sample_rate_hz = 1000000.0 / 400.0  # 2.5kHz (400µs intervals)
    t = np.linspace(0, duration_s, int(duration_s * sample_rate_hz))
    
    # 50Hz signal with small amount of noise
    signal = np.sin(2 * np.pi * 50.0 * t) + 0.05 * np.random.normal(0, 1, len(t))
    
    print(f"📊 Test Signal: {len(signal)} samples over {duration_s*1000:.1f}ms")
    print(f"⏱️  Sample Rate: {sample_rate_hz:.1f} Hz (400µs intervals)")
    print(f"🌊 Signal: 50Hz sine wave with noise")
    
    # Process all samples
    results = []
    for voltage in signal:
        result = simulator.process_voltage_sample(voltage)
        results.append(result)
    
    # Print results
    zero_crossing_results = [r for r in results if r['zero_crossing_detected']]
    
    print(f"\n🎯 Zero Crossings Detected: {len(zero_crossing_results)}")
    
    if len(zero_crossing_results) > 0:
        # Show first few zero crossings
        print("\n📈 First Zero Crossings:")
        for i, zc in enumerate(zero_crossing_results[:5]):
            time_ms = zc['time_ns'] / 1000000.0
            print(f"  {i+1}: t={time_ms:.2f}ms, f={zc['frequency_hz']:.2f}Hz, sign={'🔺' if zc['voltage_sign_positive'] else '🔻'}")
        
        # Statistics
        stats = simulator.get_statistics()
        print(f"\n📊 Frequency Measurement Statistics:")
        print(f"  Average: {stats['average_frequency']:.2f} Hz")
        print(f"  Std Dev: {stats['frequency_std']:.4f} Hz")
        print(f"  Range: {stats['min_frequency']:.2f} - {stats['max_frequency']:.2f} Hz")
        
        # Test interpolation accuracy
        last_result = zero_crossing_results[-1]
        interp = last_result['interpolation']
        print(f"\n🔍 Interpolation Example (last zero crossing):")
        print(f"  Voltage Last: {interp['voltage_last']:.6f}")
        print(f"  Voltage Current: {interp['voltage_current']:.6f}")
        print(f"  Interpolation Ratio: {interp['ratio']:.6f}")
        print(f"  Time Offset: {interp['time_offset_ns']:.1f} ns")
    
    print("\n✅ TwinCAT Zero Crossing Detector 1:1 Replica Test Complete!")