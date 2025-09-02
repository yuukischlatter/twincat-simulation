# test_zc.py
from zero_crossing_detector import ZeroCrossingSimulator
import numpy as np

print('🎯 Testing Zero Crossing...')
simulator = ZeroCrossingSimulator()

# Generate 50Hz test signal
zero_crossings_found = 0
for i in range(50):
    t = i * 0.4e-3  # 400µs intervals
    voltage = 325 * np.sin(2 * np.pi * 50 * t)
    result = simulator.process_voltage_sample(voltage)
    
    if result['zero_crossing_detected']:
        print(f'ZC {zero_crossings_found}: t={result["time_ns"]/1e6:.1f}ms, freq={result["frequency_hz"]:.2f}Hz')
        zero_crossings_found += 1

stats = simulator.get_statistics()
print(f'Total zero crossings: {zero_crossings_found}')
print(f'Stats: {stats}')

input("Press Enter to continue...")