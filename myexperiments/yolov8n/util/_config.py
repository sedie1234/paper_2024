import argparse
import yaml
import cv2
import os

def get_args():
    parser = argparse.ArgumentParser(description='option setting')
    
    parser.add_argument('--env', required=False, default='vmk180', \
                                                help='[vmk180 / alveo] (default:vmk180)')
    
    parser.add_argument('--app', required=False, default='detection', \
                                                help='[detection / segmentation] (default:detection)')
    
    parser.add_argument('--datatype', required=False, default='fp32', \
                                                help='[int8 / fp16 / fp32] (default: fp32)')
    
    parser.add_argument('--model', required=False, default='yolov8n.onnx',
                                                help='[your model]')
    
    parser.add_argument('--tool', required=False, default='ort',
                                                help='[ort / om] : onnxruntime or onnx-mlir')
    
    args = parser.parse_args()
    return args

def gst_init():
    # input_pipe = "v4l2src device=/dev/video0 ! \
    #             video/x-raw, framerate=15/1, \
    #             width=640, height=480, \
    #             format=(string)UYVY ! \
    #             videoconvert ! \
    #             appsink"

    input_pipe = "xlnxvideosrc src-type=usbcam io-mode=mmap ! \
                video/x-raw, framerate=15/1, \
                width=640, height=480, \
                format=(string)UYVY ! \
                videoconvert ! \
                appsink"

    capture = cv2.VideoCapture(input_pipe, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        print("Could not open webcam")
        exit()

    # width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width = round(1600)
    height = round(1080)

    fps = capture.get(cv2.CAP_PROP_FPS)

    # output_pipe = "appsrc ! \
    #             videoconvert ! \
    #             autovideosink sync=false"

    output_pipe = "appsrc ! \
                decodebin ! \
                videoconvert ! \
                autovideosink sync=false"
                # kmssink driver-name=xlnx plane-id=39 render-rectangle=<0,0,1920,1080>"
                

    writer = cv2.VideoWriter(output_pipe, 0, fps, (width, height), 1)
    if not writer.isOpened():
        print("Could not open streamer")
        exit()
    
    return capture, writer

def x86_init():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open webcam")
        exit()
    
    fps = capture.get(cv2.CAP_PROP_FPS)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return capture

def get_classes(app_type):
    cwd = os.getcwd()
    if app_type == 'detection':
        yaml_path = cwd + '/obj_class/coco128.yaml'
    else:
        yaml_path = cwd + '/obj_class/coco8-seg.yaml'
    
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        classes = data['names']
    
    return classes
    