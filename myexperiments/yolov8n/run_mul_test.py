import numpy as np
from PyRuntime import OMExecutionSession

# 모델 세션 생성
session = OMExecutionSession('./mul_test_model.so')

# 입력 데이터 생성 (1x16x320x320 크기의 랜덤 데이터)
input_data_1 = np.random.rand(1, 16, 320, 320).astype(np.float32)
input_data_2 = np.random.rand(1, 16, 320, 320).astype(np.float32)

print("===Multiplication Test===")
print("Session run!")

# 모델 실행 (입력 데이터를 사용하여 Mul 연산 수행)
outputs = session.run([input_data_1, input_data_2])
print("Session run END")
output_data = outputs[0]

# 예상 출력 계산 (Mul 연산의 예상 결과는 입력 데이터들의 element-wise 곱)
expected_output_data = input_data_1 * input_data_2

print("Input Data 1:")
print(input_data_1)

print("\nInput Data 2:")
print(input_data_2)

print("\nOutput Data from Model:")
print(output_data)

print("\nExpected Output Data:")
print(expected_output_data)

# 결과 비교
comparison = np.allclose(output_data, expected_output_data, atol=1e-6)
print(f"\nModel output matches expected output: {comparison}")

# 데이터 저장 (원하면 활성화)
# np.save('input_data_1.npy', input_data_1)
# np.save('input_data_2.npy', input_data_2)
# np.save('output_data.npy', output_data)
# np.save('expected_output_data.npy', expected_output_data)

print("\nInput, output, and expected output data have been saved as 'input_data_1.npy', 'input_data_2.npy', 'output_data.npy', and 'expected_output_data.npy'")
