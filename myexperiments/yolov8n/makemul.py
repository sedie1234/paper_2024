import onnx
from onnx import helper, TensorProto

# 모델에서 입력 텐서의 shape를 1x16x320x320으로 설정
input_shape = [1, 16, 320, 320]

# 첫 번째 입력 텐서를 생성합니다
input_tensor_1 = helper.make_tensor_value_info('input_1', TensorProto.FLOAT, input_shape)

# 두 번째 입력 텐서를 생성합니다 (첫 번째 입력과 동일한 shape으로)
input_tensor_2 = helper.make_tensor_value_info('input_2', TensorProto.FLOAT, input_shape)

# Mul 노드 생성
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
mul_node = helper.make_node(
    'Mul',
    inputs=[input_tensor_1.name, input_tensor_2.name],  # 두 개의 입력을 사용
    outputs=['output'],
    name='MulNode'
)

# 그래프 정의
graph = helper.make_graph(
    nodes=[mul_node],
    name='MulGraph',
    inputs=[input_tensor_1, input_tensor_2],  # 두 개의 입력 정의
    outputs=[output_tensor],
)

# 모델 생성
new_model = helper.make_model(graph, producer_name='mul-model-example')
onnx.checker.check_model(new_model)

# 모델 저장
output_model_path = './mul_test_model.onnx'
onnx.save(new_model, output_model_path)

print(f"Mul ONNX 모델이 성공적으로 생성되었습니다: {output_model_path}")
