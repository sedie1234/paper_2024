import onnxruntime
import json
import cv2
import math
import numpy as np
from PIL import Image
from PyRuntime import OMExecutionSession
import time

from . _utils import xywh2xyxy, nms, draw, sigmoid, rescale_boxes
from . _config import get_classes


control_session = 2 # 2: compare onnxruntime & onnx-mlir // 1: onnxruntime // 0: onnx-mlir
model_split = 0
ort_on = 1
om_on = 1

class Yolo:
    def __init__(self, data_type, app_type, conf_thres=0.3, iou_thres=0.5, input_size=640):

        self.classes = get_classes(app_type)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.data_type = data_type
        self.app_type = app_type

        self.num_masks = 32

        self.session = None
        self.sessions = []
        if control_session == 1:
            if model_split == 0:
                self.session = self.sess_init("yolov8n.onnx")
            elif model_split == 1:
                self.sessions.append(self.sess_init('./part1.onnx'))
                self.sessions.append(self.sess_init('./part2.onnx'))
        elif control_session == 0:
            if model_split == 0:
                self.session = self.sess_init('./yolov8n.so')
            elif model_split == 1:
                self.sessions.append(self.sess_init('./part1.so'))
                self.sessions.append(self.sess_init('./part2.so'))
        elif control_session == 2:
            self.session = onnxruntime.InferenceSession('./_9_part1.onnx')
            self.sessions.append(OMExecutionSession('./_9_part1.so'))
                
        
        self.input_size = input_size

        self.cap_width, self.cap_height = [0, 0]

        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

    def sess_init(self, model):
        session = None
        if control_session:
			#model_path = 'conv.onnx'
            session = onnxruntime.InferenceSession(model,
                                            providers=['CPUExecutionProvider'])
        else:
            session = OMExecutionSession(model)
			#session = OMExecutionSession('./conv.so')

        return session

    def preprocess(self, image):
        self.cap_width, self.cap_height = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_size, self.input_size))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]

        #return input_tensor.astype(np.float32)
        return input_tensor.astype(np.float32)

    def inference(self, image):
        input_tensor = self.preprocess(image)
        temp_out = input_tensor
        outputs = None
        if control_session == 1:
            if model_split == 0:
                output_names = [self.session.get_outputs()[i].name for i in range(len(self.session.get_outputs()))]
                outputs = self.session.run(output_names=output_names, input_feed={self.session.get_inputs()[0].name: input_tensor})
            elif model_split == 1:
                for sess in self.sessions:
                    input = temp_out
                    if(isinstance(input, list)):
                        input = np.array(input)
                        input = input[0]
                    
                    temp_out_names= [sess.get_outputs()[i].name for i in range(len(sess.get_outputs()))]
                    temp_out = sess.run(output_names=temp_out_names, input_feed={sess.get_inputs()[0].name: input})
                    print('[temp_out]')
                    print(temp_out)
                    
                outputs = temp_out
        elif control_session == 0:
            print(input_tensor)
            temp_out = input_tensor
            if model_split == 0:
                outputs = self.session.run(input_tensor)
            elif model_split == 1:
                for sess in self.sessions:
                    input = temp_out
                    if(isinstance(input, list)):
                        input = np.array(input)
                        input = input[0]
                    temp_out = sess.run(input)
                    
                outputs = temp_out
        elif control_session == 2:
            print(input_tensor.shape)
            print(input_tensor)
            temp_out = input_tensor
            if ort_on == 1:
                output_names = [self.session.get_outputs()[i].name for i in range(len(self.session.get_outputs()))]
                ort_output = self.session.run(output_names=output_names, input_feed={self.session.get_inputs()[0].name: input_tensor})
                outputs = ort_output
                np.save('ortout.npy', ort_output)


            if om_on == 1:
                om_output = self.sessions[0].run(input_tensor)
                outputs = om_output
                np.save('omout.npy', om_output)
                
            if om_on == 1 and ort_on == 1:
                comparison = np.allclose(ort_output, om_output, atol=1e-6)
                print(f"\nModel output matches expected output: {comparison}")
                
                print("[ort output]")
                print(ort_output)
                print("[om output]")
                print(om_output)
            
        return outputs

    def post_process(self, outputs):
        if self.app_type == 'detection':
            boxes, scores, class_ids, mask_pred = self.box_process(outputs[0])
            return boxes, scores, class_ids
        else:
            self.boxes, scores, class_ids, mask_pred = self.box_process(outputs[0])

        mask_maps = self.mask_process(mask_pred, outputs[1])

        return self.boxes, scores, class_ids, mask_maps

    def box_process(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        if self.app_type == 'segmentation':
            scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        else:
            scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def mask_process(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))
        scale_boxes = rescale_boxes(self.boxes,
                                   (self.cap_width, self.cap_height),
                                   (mask_width, mask_height))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.cap_width, self.cap_height))
        blur_size = (int(self.cap_width / mask_width), int(self.cap_height / mask_height))

        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]

            try:
                crop_mask = cv2.resize(scale_crop_mask,
                                    (x2 - x1, y2 - y1),
                                    interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                continue

            try:
                crop_mask = cv2.blur(crop_mask, blur_size)
                crop_mask = (crop_mask > 0.5).astype(np.uint8)
            except Exception as e:
                continue

            try:
                mask_maps[i, y1:y2, x1:x2] = crop_mask
            except Exception as e:
                pass

        return mask_maps

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        input_shape = [self.input_size, self.input_size]
        image_shape = [self.cap_width, self.cap_height]

        boxes = rescale_boxes(boxes, input_shape, image_shape)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def main(self, image):
        if self.app_type == 'detection':
            outputs = self.inference(image)
            boxes, scores, class_ids = self.post_process(outputs)
            result_img = draw(image, boxes, scores, class_ids, self.colors, None, 1, self.classes)
            return result_img
        elif self.app_type == 'test':
            outputs = self.inference(image)
            return outputs
        else:
            outputs = self.inference(image)
            boxes, scores, class_ids, mask_maps = self.post_process(outputs)
            result_img = draw(image, boxes, scores, class_ids, self.colors, mask_maps, 0.3, self.classes)
            return result_img
