import onnx
from onnx import helper, TensorProto

# 주어진 ONNX 모델 로드
model_path = './yolov8n.onnx'
model = onnx.load(model_path)

# Conv_371 노드 찾기
conv_node = None
for node in model.graph.node:
    if node.name == '/model.22/dfl/conv/Conv':
        conv_node = node
        break

if conv_node is None:
    raise ValueError("Conv_371 노드를 찾을 수 없습니다.")

# Conv_371 노드의 입력과 출력을 얻습니다.
input_name = conv_node.input[0]
output_name = conv_node.output[0]

# 입력 텐서의 정보를 가져옵니다
input_tensor_info = next((tensor for tensor in model.graph.value_info if tensor.name == input_name), None)
if input_tensor_info is None:
    input_tensor_info = next((tensor for tensor in model.graph.input if tensor.name == input_name), None)

input_shape = [dim.dim_value for dim in input_tensor_info.type.tensor_type.shape.dim]
input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)

# 출력 텐서의 정보를 가져옵니다
output_tensor_info = next((tensor for tensor in model.graph.value_info if tensor.name == output_name), None)
if output_tensor_info is None:
    output_tensor_info = next((tensor for tensor in model.graph.output if tensor.name == output_name), None)

output_shape = [dim.dim_value for dim in output_tensor_info.type.tensor_type.shape.dim]
output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

# Conv_371 노드와 관련된 초기화 데이터를 가져옵니다.
initializers = [init for init in model.graph.initializer if init.name in conv_node.input]

# 새로운 그래프 생성
new_graph = helper.make_graph(
    nodes=[conv_node],
    name='Conv_371_Graph',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=initializers
)

# 새로운 모델 생성
new_model = helper.make_model(new_graph, producer_name='conv_371_model')
onnx.checker.check_model(new_model)

# 새로운 모델 저장
output_model_path = './conv_371_test_model.onnx'
onnx.save(new_model, output_model_path)

print(f"Conv_371 노드만 포함된 새로운 ONNX 모델이 성공적으로 생성되었습니다: {output_model_path}")
