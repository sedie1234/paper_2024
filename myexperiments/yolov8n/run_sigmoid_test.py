import numpy as np
from PyRuntime import OMExecutionSession

session = OMExecutionSession('./sigmoid_test_model.so')

input_data = np.random.rand(1, 16, 320, 320).astype(np.float32)

print("===Sigmoid Test===")
print("session run!")
      
outputs = session.run([input_data])
print("session run END")
output_data = outputs[0]

expected_output_data = 1 / (1 + np.exp(-input_data))

print("Input Data:")
print(input_data)

print("\nOutput Data from Model:")
print(output_data)

print("\nExpected Output Data:")
print(expected_output_data)

comparison = np.allclose(output_data, expected_output_data, atol=1e-6)
print(f"\nModel output matches expected output: {comparison}")

print("\nInput, output, and expected output data have been saved as 'input_data.npy', 'output_data.npy', and 'expected_output_data.npy'")
