//+------------------------------------------------------------------+
//|                                                     ONNX.mqh |
//|                    Professional ONNX Runtime Interface          |
//+------------------------------------------------------------------+
#property copyright "ML Entry Validator Project"
#property link      ""

//--- Import functions from ONNXRuntime.dll
#import "ONNXRuntime.dll"
   long    CreateSession(string model_path);
   bool    RunInference(long session_handle, double &input_data[], double &output_data[]);
   void    DestroySession(long session_handle);
   int     GetInputCount(long session_handle);
   int     GetOutputCount(long session_handle);
   string  GetLastError();
#import

//+------------------------------------------------------------------+
//| ONNX Runtime Management Class                                    |
//+------------------------------------------------------------------+
class CONNXRuntime
{
private:
   long     m_sessionHandle;
   string   m_modelPath;
   int      m_inputCount;
   int      m_outputCount;
   bool     m_isInitialized;
   
public:
   //+------------------------------------------------------------------+
   //| Constructor                                                      |
   //+------------------------------------------------------------------+
   CONNXRuntime()
   {
      m_sessionHandle = 0;
      m_modelPath = "";
      m_inputCount = 0;
      m_outputCount = 0;
      m_isInitialized = false;
   }
   
   //+------------------------------------------------------------------+
   //| Destructor                                                       |
   //+------------------------------------------------------------------+
   ~CONNXRuntime()
   {
      if(m_sessionHandle != 0)
      {
         DestroySession(m_sessionHandle);
         m_sessionHandle = 0;
      }
   }
   
   //+------------------------------------------------------------------+
   //| Load ONNX model                                                  |
   //+------------------------------------------------------------------+
   bool Load(string model_path)
   {
      m_modelPath = model_path;
      
      // Create ONNX session
      m_sessionHandle = CreateSession(m_modelPath);
      if(m_sessionHandle == 0)
      {
         Print("ONNX ERROR: Failed to create session for model: ", m_modelPath);
         Print("ONNX ERROR: ", GetLastError());
         return false;
      }
      
      // Get model dimensions
      m_inputCount = GetInputCount(m_sessionHandle);
      m_outputCount = GetOutputCount(m_sessionHandle);
      
      if(m_inputCount <= 0 || m_outputCount <= 0)
      {
         Print("ONNX ERROR: Invalid model dimensions. Inputs: ", m_inputCount, ", Outputs: ", m_outputCount);
         DestroySession(m_sessionHandle);
         m_sessionHandle = 0;
         return false;
      }
      
      m_isInitialized = true;
      Print("ONNX SUCCESS: Model loaded successfully");
      Print("ONNX INFO: Model path: ", m_modelPath);
      Print("ONNX INFO: Input features: ", m_inputCount);
      Print("ONNX INFO: Output predictions: ", m_outputCount);
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Run model prediction                                             |
   //+------------------------------------------------------------------+
   bool Predict(double &features[], double &predictions[])
   {
      // Validation checks
      if(!m_isInitialized)
      {
         Print("ONNX ERROR: Model not initialized");
         return false;
      }
      
      if(m_sessionHandle == 0)
      {
         Print("ONNX ERROR: Invalid session handle");
         return false;
      }
      
      if(ArraySize(features) != m_inputCount)
      {
         Print("ONNX ERROR: Feature count mismatch. Expected: ", m_inputCount, ", Got: ", ArraySize(features));
         return false;
      }
      
      if(ArraySize(predictions) < m_outputCount)
      {
         Print("ONNX ERROR: Prediction array too small. Need: ", m_outputCount, ", Got: ", ArraySize(predictions));
         return false;
      }
      
      // Run inference
      bool result = RunInference(m_sessionHandle, features, predictions);
      if(!result)
      {
         Print("ONNX ERROR: Inference failed - ", GetLastError());
         return false;
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Get model information                                            |
   //+------------------------------------------------------------------+
   int GetInputCount() const { return m_inputCount; }
   int GetOutputCount() const { return m_outputCount; }
   bool IsInitialized() const { return m_isInitialized; }
   string GetModelPath() const { return m_modelPath; }
};