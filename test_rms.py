# test_rms.py
from rms_calculator import RMSMeasurementSystem
import numpy as np
import math

print('🔋 Testing RMS Calculator...')
rms_system = RMSMeasurementSystem()

# Generate test data
amplitude = 325.0
for cycle in range(10):
    time_base = cycle * 0.4e-3
    samples = []
    for i in range(8):
        t = time_base + i * 50e-6
        voltage = amplitude * math.sin(2 * math.pi * 50 * t)
        samples.append(voltage)
    
    result = rms_system.process_oversamples(samples)
    if result['rms_value'] > 0:
        print(f'Cycle {cycle}: RMS = {result["rms_value"]:.1f}V')
    else:
        print(f'Cycle {cycle}: No RMS calculated')

stats = rms_system.get_statistics()
print(f'Total RMS measurements: {stats["total_rms_measurements"]}')
print(f'Stats: {stats}')

input("Press Enter to continue...")