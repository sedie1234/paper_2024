import numpy as np
from PyRuntime import OMExecutionSession

# 모델 세션 생성
session = OMExecutionSession('./conv_test_model.so')

# 입력 데이터 생성 (1x16x320x320 크기의 랜덤 데이터)
input_data = np.random.rand(1, 64, 40, 40).astype(np.float32)

print("===Convolution Test===")
print("Session run!")

# 모델 실행 (입력 데이터를 사용하여 Convolution 연산 수행)
outputs = session.run([input_data])
print("Session run END")
output_data = outputs[0]

# 예상 출력 계산 (여기서는 간단한 Convolution 연산을 예상, 실제로는 모델 가중치 등을 고려해야 함)
# 이 부분은 예시로 실제 예상 결과는 모델의 파라미터에 따라 다를 수 있음
# 아래 코드의 예상 출력은 단순히 입력 데이터에 대해 직접적으로 Conv를 수행하는 것으로 가정한 것입니다.
# 실제로는 모델 가중치와 편향을 고려해야 합니다.
# expected_output_data = <사용자 정의 예상 결과>
# 예시로만 단순 Sigmoid 적용:
expected_output_data = np.zeros_like(output_data)  # 이 부분은 실제 예상 계산에 따라 수정

print("Input Data:")
print(input_data)

print("\nOutput Data from Model:")
print(output_data)

print("\nExpected Output Data:")
print(expected_output_data)

# 결과 비교
comparison = np.allclose(output_data, expected_output_data, atol=1e-6)
print(f"\nModel output matches expected output: {comparison}")

# 데이터 저장 (원하면 활성화)
# np.save('input_data.npy', input_data)
# np.save('output_data.npy', output_data)
# np.save('expected_output_data.npy', expected_output_data)

print("\nInput, output, and expected output data have been saved as 'input_data.npy', 'output_data.npy', and 'expected_output_data.npy'")
