# quick_fix_test.py
print("🚀 TwinCAT Quick Fix Test")

try:
    from simulation_main import TwinCATSystemSimulation, SimulationConfig
    
    # Very short test
    config = SimulationConfig(
        simulation_duration_ms=40.0,  # Only 40ms = 100 cycles
        enable_disturbances=False,    # Clean signal
        enable_harmonics=False,       # Clean signal
        mains_amplitude_v=325.0,      # Standard amplitude
        expected_l2l_voltage_rms=400.0,  # Expected L2L RMS
    )
    
    print(f"Config: {config.simulation_duration_ms}ms, {int(config.simulation_duration_ms*1000/config.measurement_task_cycle_us)} cycles")
    
    simulation = TwinCATSystemSimulation(config)
    results = simulation.run_simulation()
    
    # Analyze results
    measurement_results = results['measurement_results']
    print(f"\n📊 Analysis of {len(measurement_results)} measurements:")
    
    # Count different types of successful measurements
    zero_crossings = sum(1 for r in measurement_results if r['zero_crossing_time_ns'] > 0)
    rms_measurements = sum(1 for r in measurement_results if r['l2l_voltage_rms'] > 0)
    freq_measurements = sum(1 for r in measurement_results if r['frequency_hz'] > 0)
    
    print(f"Zero Crossings: {zero_crossings}/{len(measurement_results)} ({zero_crossings/len(measurement_results)*100:.1f}%)")
    print(f"RMS Measurements: {rms_measurements}/{len(measurement_results)} ({rms_measurements/len(measurement_results)*100:.1f}%)")
    print(f"Freq Measurements: {freq_measurements}/{len(measurement_results)} ({freq_measurements/len(measurement_results)*100:.1f}%)")
    
    # Show first few results with data
    print(f"\n🔍 First few results:")
    for i, result in enumerate(measurement_results[:10]):
        zc = "✅" if result['zero_crossing_time_ns'] > 0 else "❌"
        rms = f"{result['l2l_voltage_rms']:.1f}V" if result['l2l_voltage_rms'] > 0 else "No RMS"
        freq = f"{result['frequency_hz']:.2f}Hz" if result['frequency_hz'] > 0 else "No Freq"
        print(f"  {i}: {zc} RMS={rms}, F={freq}")
    
    # Try to identify the RMS issue
    print(f"\n🔋 RMS Debug:")
    for i, result in enumerate(measurement_results[:20]):
        if 'debug' in result and 'voltage_measurements' in result['debug']:
            rms_unfiltered = result['debug']['voltage_measurements'].get('rms_unfiltered', 0)
            l2l_rms = result['debug']['voltage_measurements'].get('l2l_voltage_rms', 0)
            print(f"  Cycle {i}: rms_unfiltered={rms_unfiltered:.1f}, l2l_rms={l2l_rms:.1f}")
            if i >= 5:  # Only show first few
                break
    
    print(f"\n✅ Quick fix test complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

input("Press Enter to continue...")