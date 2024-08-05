#!/usr/bin/env python3

import numpy as np
from PyRuntime import OMExecutionSession

# Load the model mnist.so compiled with onnx-mlir.
session = OMExecutionSession('./add.so')
# Print the models input/output signature, for display.
# Signature functions for info only, commented out if they cause problems.
print("input signature in json", session.input_signature())
print("output signature in json",session.output_signature())
# Create an input arbitrarily filled of 1.0 values.
input1 = np.array([1, 2, 3, 4, 5, 6], np.dtype(np.float32)).reshape(3,2)
input2 = np.array([7, 8, 9, 10, 11, 12], np.dtype(np.float32)).reshape(3,2)
# Run the model. It is best to always use the [] around the inputs as the inputs
# are an vector of numpy arrays.
outputs = session.run([input1, input2])
# Analyze the output (first array in the list, of signature 1x10xf32).
prediction = outputs[0]
print(prediction)
# for pred in prediction:
#     print(pred)    
# Print the value with the highest prediction (8 here).
