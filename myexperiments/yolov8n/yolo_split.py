import onnx
from onnx import helper, shape_inference, TensorProto

# Load the original model
model = onnx.load("yolov8n.onnx")

# origin_model = onnx.load("yolov8n.onnx")

# Tensor 이름을 기준으로 나눌 지점 설정
filename1 = '_3_part1.onnx'
filename2 = '_3_part2.onnx'
split_tensor = "/model.0/act/Mul_output_0"

# 모델에 대한 shape inference 실행 (출력 형상을 추론하기 위해)
inferred_model = shape_inference.infer_shapes(model)

# split_tensor의 출력 텐서 형상 가져오기
part1_output_shape = None
for value_info in inferred_model.graph.value_info:
    if value_info.name == split_tensor:
        part1_output_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        break

if part1_output_shape is None:
    raise ValueError(f"Tensor '{split_tensor}'의 형상을 가져오지 못했습니다.")

print("Split Tensor Output Shape:", part1_output_shape)

# 첫 번째 모델에 포함할 노드와 두 번째 모델에 포함할 노드 리스트 초기화
part1_nodes = []
part2_nodes = []
split_found = False

# 그래프의 모든 노드를 탐색하여 분리
for node in model.graph.node:
    # 노드의 출력 텐서에 split_tensor가 있는 경우
    if split_tensor in node.output:
        part1_nodes.append(node)
        split_found = True
    elif not split_found:
        part1_nodes.append(node)
    else:
        part2_nodes.append(node)

# 모델에서 필요한 initializer 찾기
part1_initializers = []
part2_initializers = []

# Split_157 / Split_184

# 모든 노드 확인하여 Split_157이 출력인 경우 찾기
for node in model.graph.node:
    if "onnx::Split_157" in node.output:
        print(f"Node producing 'onnx::Split_157': {node.name}")
        part2_nodes.append(node)

# for node in origin_model.graph.node:
#     if "onnx::Split_184" in node.output:
#         print(f"Node producing 'onnx::Split_184': {node.name}")
#         part2_nodes.append(node)


for node in part2_nodes:
    if "Constant_28" in node.name:
        print(node.name)
    
                
for initializer in model.graph.initializer:
    # 첫 번째 모델에 필요한 초기화 데이터
    for node in part1_nodes:
        for input in node.input:
            if input == initializer.name:
                part1_initializers.append(initializer)
    
    # 두 번째 모델에 필요한 초기화 데이터
    for node in part2_nodes:
        for input in node.input:
            if input == initializer.name:
                part2_initializers.append(initializer)

# split_tensor를 출력으로 하는 첫 번째 파트의 출력 정의
part1_output = [helper.make_tensor_value_info(split_tensor, TensorProto.FLOAT, part1_output_shape)]

# 첫 번째 모델 생성 (initializer와 value_info 포함)
part1_graph = helper.make_graph(
    part1_nodes, 
    'Part1', 
    model.graph.input, 
    part1_output, 
    initializer=part1_initializers
)

# 첫 번째 모델에 대한 shape inference 실행
part1_model = helper.make_model(part1_graph)
part1_model = shape_inference.infer_shapes(part1_model)

# 임의의 텐서 이름과 형상 설정 (part1의 출력 형상을 사용)
new_input_name = split_tensor
part2_input = [helper.make_tensor_value_info(new_input_name, TensorProto.FLOAT, part1_output_shape)]

# 두 번째 모델 생성 (initializer와 value_info 포함)
# 기존 노드들이 새 입력 텐서를 참조하도록 설정
for node in part2_nodes:
    for i, input_name in enumerate(node.input):
        if input_name == split_tensor:
            node.input[i] = new_input_name  # 기존 split_tensor 대신 새로운 입력 텐서를 참조

part2_graph = helper.make_graph(
    part2_nodes, 
    'Part2', 
    part2_input, 
    model.graph.output, 
    initializer=part2_initializers
)

# 두 번째 모델의 shape inference 실행
part2_model = helper.make_model(part2_graph)
part2_model = shape_inference.infer_shapes(part2_model)

for opset_import in part2_model.opset_import:
    opset_import.version = 19

for opset_import in part1_model.opset_import:
    opset_import.version = 19

# 첫 번째 모델과 두 번째 모델 저장
onnx.save(part1_model, filename1)
onnx.save(part2_model, filename2)
