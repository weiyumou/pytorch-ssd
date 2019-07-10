from vision.utils.misc import Timer
import cv2
import sys
import argparse
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file',
                        default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name',
                        default='pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='data/EPSON/20190214_clip/20190214_trim.mp4', type=str,
                        help='videos or image files')
    parser.add_argument('--net_type', type=str, help='model name',
                        default='mb1-ssd')
    parser.add_argument('--model_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190618/mb1-ssd-Epoch-99-Loss-2.7762672106424966.pth')
    parser.add_argument('--label_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190618/open-images-model-labels.txt')
    args = parser.parse_args()
    return args


def prepare_predictor(net_type, model_path, label_path):
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(num_classes, is_test=True)
    else:
        raise ValueError("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        raise ValueError("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

    return class_names, predictor


def do_object_detection(orig_image, predictor, class_names):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    # interval = timer.end()
    # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    # print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)


if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)  # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

args = parse_args()
cap = cv2.VideoCapture(args.video_name)
class_names, predictor = prepare_predictor(args.net_type, args.model_path, args.label_path)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    do_object_detection(orig_image, predictor, class_names)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
