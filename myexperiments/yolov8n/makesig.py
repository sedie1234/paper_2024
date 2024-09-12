import onnx
from onnx import helper, TensorProto

# 주어진 ONNX 모델 로드
model_path = './yolov8n.onnx'
model = onnx.load(model_path)

# 모델에서 첫 번째 입력 텐서를 가져옵니다
#input_tensor = model.graph.input[0]

# 첫 번째 입력 텐서의 shape 정보를 가져옵니다
input_shape = [1, 16, 320, 320]
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)

# Sigmoid 노드 생성
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
sigmoid_node = helper.make_node(
    'Sigmoid',
    inputs=[input_tensor.name],
    outputs=['output'],
    name='SigmoidNode'
)

# 그래프 정의
graph = helper.make_graph(
    nodes=[sigmoid_node],
    name='SigmoidGraph',
    inputs=[input_tensor],
    outputs=[output_tensor],
)

# 모델 생성
new_model = helper.make_model(graph, producer_name='sigmoid-model-example')
onnx.checker.check_model(new_model)

# 모델 저장
output_model_path = './sigmoid_test_model.onnx'
onnx.save(new_model, output_model_path)

print(f"Sigmoid ONNX 모델이 성공적으로 생성되었습니다: {output_model_path}")

