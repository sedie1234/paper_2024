import argparse
import onnxruntime
import cv2
import timeit

from util import _config
from util import _yolo
from util import _utils

opt = _config.get_args()
classes = _config.get_classes(opt.app)

if __name__ == '__main__':
    
    yolo = _yolo.Yolo(opt.datatype, opt.app, opt.model)
    counter = 0

    # start time
    start_t = timeit.default_timer()

    frame = cv2.imread("images/test.jpg")
    result_img = yolo.main(frame)
    #cv2.imshow("Objects {}".format(opt.a), result_img)
    # cv2.imwrite('results/test_out.jpg', result_img)
    
    # end time
    # terminate_t = timeit.default_timer()

    # fps = float(1./(terminate_t - start_t ))
    # print("fps:{}\n".format(round(fps,1)))



    