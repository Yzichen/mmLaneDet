from argparse import ArgumentParser
import cv2
import mmcv
from mmlane.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='demo.jpg', help='Image file')
    parser.add_argument('--config', default=None, help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument('--show', default=False, help='Show result')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result(
        model,
        args.img,
        result,
        score_thr=args.score_thr,
        show=args.show,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)