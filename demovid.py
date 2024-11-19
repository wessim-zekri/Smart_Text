import string
import argparse
import cv2  # Import OpenCV for video processing

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from datasetvid import AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_frame(model, converter, frame, opt):
    # Preprocess the frame
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    image_tensors = AlignCollate_demo([frame])

    # Convert to tensor and move to GPU if available
    image = image_tensors[0].unsqueeze(0).to(device)  # Add batch dimension

    # Prepare input for prediction
    batch_size = image.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    # Make prediction
    if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred)

        # Greedy decoding to get predicted text
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)
    else:
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    pred = preds_str[0]
    pred_max_prob = preds_max_prob[0]

    # Post-processing
    if 'Attn' in opt.Prediction:
        pred_EOS = pred.find('[s]')
        pred = pred[:pred_EOS]  # Prune after "end of sentence" token ([s])
        pred_max_prob = pred_max_prob[:pred_EOS]

    # Calculate confidence score
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

    return pred, confidence_score


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # Open the video file
    video_capture = cv2.VideoCapture(opt.video_path)

    if not video_capture.isOpened():
        print(f"Error opening video file: {opt.video_path}")
        return

    # Process video frame by frame
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1

        # Convert frame to RGB if necessary
        if not opt.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform text recognition on the current frame
        pred, confidence_score = process_frame(model, converter, frame, opt)

        # Display the result
        print(f'Frame {frame_count}: Predicted Text: {pred} | Confidence Score: {confidence_score:.4f}')

    # Release the video capture object
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, help='path to input video')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
