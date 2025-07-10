#--- Complete Pipeline Execution Script ---
import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"✓ SUCCESS: {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: {description} failed")
        print(f"Return code: {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ EXCEPTION: {str(e)}")
        return False

def main():
    """Execute the complete ML Entry Validator pipeline"""
    print("="*80)
    print("HYBRID ML ENTRY VALIDATOR EA - COMPLETE PIPELINE EXECUTION")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Pipeline steps
    pipeline_steps = [
        ("1_data_acquisition.py", "Data Acquisition from MT5"),
        ("2_feature_engineering.py", "Feature Engineering & Context Creation"),
        ("3_labeling.py", "Simulation-Based Trade Labeling"),
        ("4_model_training.py", "Baseline Model Training & Validation"),
        ("5_genetic_optimization.py", "Genetic Algorithm Hyperparameter Optimization"),
        ("6_onnx_export.py", "Final Model Export to ONNX Format")
    ]
    
    # Track execution results
    results = []
    
    # Execute each step
    for script, description in pipeline_steps:
        success = run_script(script, description)
        results.append((script, description, success))
        
        if not success:
            print(f"\n⚠️  PIPELINE STOPPED: {description} failed")
            print("Please check the error messages above and fix issues before continuing.")
            break
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    for script, description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    # Check if all steps completed
    all_success = all(result[2] for result in results)
    
    if all_success:
        print("\n🎉 COMPLETE SUCCESS: All pipeline steps executed successfully!")
        print("\nGenerated Artifacts:")
        print("📁 data/")
        print("   ├── raw_data.csv (Historical OHLCV data)")
        print("   ├── featured_data.csv (Engineered features)")
        print("   └── final_dataset.csv (Labeled golden dataset)")
        print("📁 models/")
        print("   ├── entry_model.onnx (Deployable ML model)")
        print("   ├── model_info.json (Model metadata)")
        print("   ├── optimized_params.json (Best hyperparameters)")
        print("   └── final_model_metadata.json (Deployment info)")
        print("📄 feature_schema.md (Feature documentation)")
        print("📄 ONNX.mqh (MQL5 interface header)")
        print("📄 ML_Entry_Validator.mq5 (Complete Expert Advisor)")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Copy ONNX.mqh to your MQL5/Include directory")
        print("2. Copy ML_Entry_Validator.mq5 to your MQL5/Experts directory")
        print("3. Copy entry_model.onnx to your MQL5/Files directory")
        print("4. Obtain and install ONNXRuntime.dll in MQL5/Libraries")
        print("5. Compile and test the Expert Advisor in MetaTrader 5")
        
    else:
        failed_steps = [result[1] for result in results if not result[2]]
        print(f"\n❌ PIPELINE INCOMPLETE: {len(failed_steps)} step(s) failed")
        print("Failed steps:")
        for step in failed_steps:
            print(f"   - {step}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("="*80)

if __name__ == "__main__":
    main()