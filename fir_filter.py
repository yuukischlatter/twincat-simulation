"""
fir_filter.py - TwinCAT FB_FIRFilterOvSampl 1:1 Replica
=====================================================

Original TwinCAT Source: FB_FIRFilterOvSampl.TcPOU (Document 12)
Company: Schlatter Industries AG
Application: SWEP 30 Welding System

This is an exact 1:1 replica of the TwinCAT FIR filter used for 
50Hz mains voltage filtering in industrial welding applications.
"""

import numpy as np
from typing import List, Optional


class FB_FIRFilterOvSampl:
    """
    1:1 Replica of TwinCAT FB_FIRFilterOvSampl
    
    Original TwinCAT Constants:
    - FILTER_ORDER: INT := 136
    - OVERSAMPLING: INT := 8
    
    Used in FB_EL3783_LxLy for filtering mains voltage oversamples
    with 136th order FIR filter for precise zero-crossing detection.
    """
    
    # TwinCAT Constants - exact 1:1
    FILTER_ORDER = 136          # Original: FILTER_ORDER : INT := 136
    OVERSAMPLING = 8           # Original: OVERSAMPLING : INT := 8
    
    def __init__(self):
        """
        Initialize FB_FIRFilterOvSampl instance
        
        TwinCAT VAR section replica:
        - b: ARRAY[0..FILTER_ORDER] OF LREAL - Filter coefficients
        - circularBuffer: ARRAY[0..FILTER_ORDER] OF LREAL - Ring buffer
        - writePointer: INT - Array element pointer for next input
        - phaseShiftP1: LREAL - Phase shift at frequencyP1 in [°]
        - frequencyP1: LREAL - Frequency at phaseShiftP1 in [Hz]
        - bConfigured: BOOL - Flag that filter is configured
        """
        # Filter coefficients (b[0..136])
        self.b = np.zeros(self.FILTER_ORDER + 1, dtype=np.float64)
        
        # Circular buffer for filter intermediate values
        self.circularBuffer = np.zeros(self.FILTER_ORDER + 1, dtype=np.float64)
        
        # Write pointer for next input value
        self.writePointer = 0
        
        # Phase shift at frequencyP1 in [°]
        self.phaseShiftP1 = 0.0
        
        # Frequency at phaseShift in [Hz]
        self.frequencyP1 = 0.0
        
        # Configuration flag
        self.bConfigured = False
        
        # Source info for error logging (TwinCAT equivalent)
        self.fbSourceInfo_sName = "FB_FIRFilterOvSampl"
    
    def Configure(self, aCoefficients: List[float], phaseShift: float, frequency: float) -> bool:
        """
        Configure FIR filter parameters
        
        TwinCAT Method: Configure : BOOL
        
        Args:
            aCoefficients: Filter coefficients array [0..136]
            phaseShift: Phase shift at "frequency" in [°]  
            frequency: Frequency at "phaseShift" in [Hz]
            
        Returns:
            bool: True if configured successfully, False otherwise
            
        Original TwinCAT Logic:
        - Array bounds validation: (lowerBound = 0) AND (upperBound = FILTER_ORDER)
        - Store coefficients in b[idx] array
        - Set phase shift and frequency parameters
        - Log error if configuration fails
        """
        # Array bounds determination - TwinCAT: LOWER_BOUND/UPPER_BOUND
        lowerBound = 0
        upperBound = len(aCoefficients) - 1
        
        # Array bounds check - exact TwinCAT condition
        if lowerBound == 0 and upperBound == self.FILTER_ORDER:
            # Store parameters - TwinCAT: FOR idx := 0 TO FILTER_ORDER DO
            for idx in range(self.FILTER_ORDER + 1):
                self.b[idx] = aCoefficients[idx]
            
            # Store phase shift and frequency
            self.phaseShiftP1 = phaseShift
            self.frequencyP1 = frequency
            self.bConfigured = True
        else:
            self.bConfigured = False
            # TwinCAT error logging equivalent
            print(f"ERROR: FIR filter not configured - {self.fbSourceInfo_sName}")
            print(f"Expected coefficient array size: {self.FILTER_ORDER + 1}, got: {len(aCoefficients)}")
        
        # Return value assignment - TwinCAT: Configure := bConfigured
        return self.bConfigured
    
    def Call(self, newSample: List[float]) -> float:
        """
        Filter function call for oversamples
        
        TwinCAT Method: Call : LREAL
        
        Args:
            newSample: Array of oversamples [0..7] (8 samples)
            
        Returns:
            float: Filtered sample value (last filtered oversample)
            
        Original TwinCAT Logic:
        - Validate array bounds: lowerBound=0, upperBound=OVERSAMPLING-1
        - Apply filter to all oversamples sequentially
        - Update circular buffer and write pointer for each sample
        - Calculate filtered output using convolution
        """
        # Array bounds determination - TwinCAT equivalent
        lowerBound = 0
        upperBound = len(newSample) - 1
        
        # Configuration and array bounds check
        if self.bConfigured and lowerBound == 0 and upperBound == (self.OVERSAMPLING - 1):
            filteredSample = 0.0
            
            # Apply filter to all oversamples - TwinCAT: FOR idxOverSample := 0 TO (OVERSAMPLING-1)
            for idxOverSample in range(self.OVERSAMPLING):
                # Write new input value to buffer
                self.circularBuffer[self.writePointer] = newSample[idxOverSample]
                
                # Increment pointer modulo buffer size
                # TwinCAT: writePointer := (writePointer + 1) MOD (FILTER_ORDER + 1)
                self.writePointer = (self.writePointer + 1) % (self.FILTER_ORDER + 1)
                
                # Calculate new output value - TwinCAT convolution loop
                filteredSample = 0.0
                for idxCoeff in range(self.FILTER_ORDER + 1):
                    buffer_idx = (self.writePointer + idxCoeff) % (self.FILTER_ORDER + 1)
                    filteredSample += self.b[idxCoeff] * self.circularBuffer[buffer_idx]
            
            # Return last filtered sample (TwinCAT: call := filteredSample)
            return filteredSample
        else:
            # Return 0 if not configured or wrong array size
            return 0.0
    
    def GetPhaseShift(self, frequency: float) -> float:
        """
        Get phase shift in [°] for specified frequency
        
        TwinCAT Method: GetPhaseShift : LREAL
        
        Args:
            frequency: Base frequency in [Hz] (range 0..100Hz)
            
        Returns:
            float: Phase shift in [°] for specified frequency
            
        Original TwinCAT Logic:
        - Calculate phase shift for given frequency
        - Frequency range validation: frequency < 100Hz
        - Linear interpolation based on frequencyP1 and phaseShiftP1
        """
        # Phase shift calculation for given frequency
        # TwinCAT: IF (frequencyP1 <> 0) AND (frequency < 100) THEN
        if self.frequencyP1 != 0 and frequency < 100:
            # TwinCAT: GetPhaseShift := (frequency / frequencyP1) * phaseShiftP1
            return (frequency / self.frequencyP1) * self.phaseShiftP1
        else:
            # TwinCAT: GetPhaseShift := 0.0
            return 0.0


def create_twincat_coefficients() -> List[float]:
    """
    Create the exact 137 FIR coefficients from TwinCAT FB_EL3783_LxLy
    
    These coefficients are from the original TwinCAT code in FB_EL3783_LxLy.Update()
    method where they are hardcoded as coefficients[0] to coefficients[136].
    
    Filter specifications:
    - 136th order FIR filter (137 coefficients)
    - Designed for 50Hz mains frequency filtering
    - Phase shift: -122.4° at 100Hz
    - Symmetric coefficients (linear phase)
    
    Returns:
        List[float]: Exact 137 coefficients from TwinCAT
    """
    # Original TwinCAT coefficients - exact copy from FB_EL3783_LxLy
    coefficients = [
        0.000966676535800659, 0.000976699850899367, 0.000998671173490568, 0.001032696613065560,
        0.001078852319481080, 0.001137184082054750, 0.001207707008144050, 0.001290405282505980,
        0.001385232008510400, 0.001492109132051520, 0.001610927448770090, 0.001741546694964790,
        0.001883795722334440, 0.002037472756455540, 0.002202345738660960, 0.002378152750747800,
        0.002564602521705090, 0.002761375015416230, 0.002968122098057840, 0.003184468283685760,
        0.003410011556272600, 0.003644324266238130, 0.003886954099296610, 0.004137425115233080,
        0.004395238854014830, 0.004659875506445800, 0.004930795146380150, 0.005207439021328260,
        0.005489230898113790, 0.005775578460075170, 0.006065874752149330, 0.006359499670029930,
        0.006655821489457460, 0.006954198431575180, 0.007253980260171960, 0.007554509906533100,
        0.007855125117531370, 0.008155160122514580, 0.008453947314482770, 0.008750818940997220,
        0.009045108800226160, 0.009336153937507620, 0.009623296337798810, 0.009905884609383340,
        0.010183275654223300, 0.010454836320372100, 0.010719945031905200, 0.010977993391882600,
        0.011228387753922700, 0.011470550758050600, 0.011703922826575200, 0.011927963615856900,
        0.012142153419943400, 0.012345994522184500, 0.012539012491071900, 0.012720757416707000,
        0.012890805084459400, 0.013048758082550200, 0.013194246840478900, 0.013326930595399700,
        0.013446498283755700, 0.013552669355683900, 0.013645194509919500, 0.013723856347148500,
        0.013788469939985100, 0.013838883317981400, 0.013874977866315500, 0.013896668637043400,
        0.013903904572045300,  # Center coefficient [68] - maximum value
        # Symmetric part (mirror of first half)
        0.013896668637043400, 0.013874977866315500, 0.013838883317981400, 0.013788469939985100,
        0.013723856347148500, 0.013645194509919500, 0.013552669355683900, 0.013446498283755700,
        0.013326930595399700, 0.013194246840478900, 0.013048758082550200, 0.012890805084459400,
        0.012720757416707000, 0.012539012491071900, 0.012345994522184500, 0.012142153419943400,
        0.011927963615856800, 0.011703922826575200, 0.011470550758050600, 0.011228387753922700,
        0.010977993391882600, 0.010719945031905200, 0.010454836320372100, 0.010183275654223300,
        0.009905884609383350, 0.009623296337798810, 0.009336153937507620, 0.009045108800226160,
        0.008750818940997220, 0.008453947314482770, 0.008155160122514580, 0.007855125117531370,
        0.007554509906533110, 0.007253980260171960, 0.006954198431575180, 0.006655821489457460,
        0.006359499670029930, 0.006065874752149330, 0.005775578460075170, 0.005489230898113790,
        0.005207439021328270, 0.004930795146380150, 0.004659875506445800, 0.004395238854014840,
        0.004137425115233080, 0.003886954099296620, 0.003644324266238130, 0.003410011556272600,
        0.003184468283685760, 0.002968122098057840, 0.002761375015416230, 0.002564602521705090,
        0.002378152750747800, 0.002202345738660960, 0.002037472756455540, 0.001883795722334440,
        0.001741546694964790, 0.001610927448770090, 0.001492109132051520, 0.001385232008510400,
        0.001290405282505980, 0.001207707008144050, 0.001137184082054750, 0.001078852319481080,
        0.001032696613065560, 0.000998671173490567, 0.000976699850899367, 0.000966676535800659
    ]
    
    # Verify coefficient count (should be 137)
    assert len(coefficients) == 137, f"Expected 137 coefficients, got {len(coefficients)}"
    
    return coefficients


def create_configured_filter() -> FB_FIRFilterOvSampl:
    """
    Create and configure FIR filter with original TwinCAT parameters
    
    TwinCAT configuration from FB_EL3783_LxLy.Update():
    - coefficients: 137 hardcoded values
    - phaseShift: -122.4° 
    - frequency: 100.0 Hz
    
    Returns:
        FB_FIRFilterOvSampl: Configured filter ready for use
    """
    # Create filter instance
    fir_filter = FB_FIRFilterOvSampl()
    
    # Get TwinCAT coefficients
    coefficients = create_twincat_coefficients()
    
    # Configure with original TwinCAT parameters
    # TwinCAT: bFirFilterConfigured := firFilter.Configure(coefficients, phaseShift := -122.4, frequency := 100.0)
    success = fir_filter.Configure(coefficients, -122.4, 100.0)
    
    if success:
        print("✅ FIR Filter configured successfully with TwinCAT parameters")
        print(f"   - Filter Order: {fir_filter.FILTER_ORDER}")
        print(f"   - Oversampling: {fir_filter.OVERSAMPLING}")
        print(f"   - Phase Shift: {fir_filter.phaseShiftP1}° at {fir_filter.frequencyP1}Hz")
    else:
        print("❌ FIR Filter configuration failed!")
    
    return fir_filter


if __name__ == "__main__":
    """
    Test the FIR filter with sample data
    """
    print("🔧 TwinCAT FB_FIRFilterOvSampl Test")
    print("=" * 40)
    
    # Create configured filter
    fir_filter = create_configured_filter()
    
    # Test with 8 oversamples (simulating 50µs intervals)
    test_samples = [0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1, -0.1]
    
    print(f"\n📊 Test Input (8 oversamples): {test_samples}")
    
    # Filter the samples
    filtered_result = fir_filter.Call(test_samples)
    print(f"🔄 Filtered Output: {filtered_result:.8f}")
    
    # Test phase shift calculation for 50Hz
    phase_50hz = fir_filter.GetPhaseShift(50.0)
    print(f"📐 Phase Shift at 50Hz: {phase_50hz:.2f}°")
    
    print("\n✅ TwinCAT FIR Filter 1:1 Replica Test Complete!")