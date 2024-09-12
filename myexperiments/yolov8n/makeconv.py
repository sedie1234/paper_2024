import onnx
from onnx import helper, TensorProto

# 기존 모델 로드
model_path = "./yolov8n.onnx"
model = onnx.load(model_path)

# 첫 번째 Conv 노드 찾기
first_conv_node = None
# for node in model.graph.node:
#     if node.op_type == 'Conv':
#         first_conv_node = node
#         break

for node in model.graph.node:
    if node.name == "/model.6/m.0/cv1/conv/Conv":
        first_conv_node = node
        break

if first_conv_node is None:
    raise ValueError("No Conv node found in the model")

# 첫 번째 Conv 노드의 입력과 출력을 얻습니다.
input_name = first_conv_node.input[0]
output_name = first_conv_node.output[0]

# 입력 텐서의 타입 및 shape 정보를 가져옵니다.
input_shape = [1, 64, 40, 40]
input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)

# 출력 텐서의 shape를 1x16x320x320으로 설정합니다.
output_shape = [1, 64, 40, 40]
output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

# 새로운 그래프를 생성하고, 첫 번째 Conv 노드를 추가합니다.
new_graph = helper.make_graph(
    nodes=[first_conv_node],
    name='FirstConvGraph',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[init for init in model.graph.initializer if init.name in first_conv_node.input]
)

# 새로운 모델을 생성합니다.
new_model = helper.make_model(new_graph, producer_name='first-conv-model')
onnx.checker.check_model(new_model)

# 새로운 모델을 저장합니다.
new_model_path = "./conv_test_model.onnx"
onnx.save(new_model, new_model_path)

print(f"출력 크기가 1x16x320x320인 첫 번째 Conv 노드만 포함된 새로운 ONNX 모델이 생성되었습니다: {new_model_path}")
