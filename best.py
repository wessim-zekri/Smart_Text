import numpy as np
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from paddleocr import PaddleOCR
from PIL import Image
from langdetect import detect, LangDetectException
from bidi.algorithm import get_display
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

from llama import createModel, generateTrans
from ttsg import txt2spch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_paddleocr_instance(language):
    lang_map = {
        'ar': 'arabic',
        'en': 'en',
        'fr': 'fr',
        'zh': 'ch',
        'de': 'de',
        'ko': 'korean',
        'ja': 'japan',
        'it': 'latin',
        'bn': 'devanagari',
        'hi': 'devanagari'
    }
    lang_code = lang_map.get(language, 'en')  # Default to English if language not found
    return PaddleOCR(use_angle_cls=True, lang=lang_code)

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
    model = model.to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"Best_Label":25s}\t{"Confidence_Score":25s}'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score_str = pred_max_prob.cumprod(dim=0)[-1].item()

                # Detect language from STR result
                try:
                    detected_language = detect(pred)
                except LangDetectException:
                    detected_language = 'en'  # Fallback to English if detection fails
                print(f'Detected language for {img_name}: {detected_language}')

                # Initialize PaddleOCR with detected language
                ocr = get_paddleocr_instance(detected_language)

                # Read image using PIL and convert to numpy array
                image = Image.open(img_name).convert('RGB')
                image_np = np.array(image)

                # PaddleOCR prediction
                ocr_result = ocr.ocr(np.array(image_np), cls=True)
                print(f'OCR result for {img_name}: {ocr_result}')  # Debugging line

                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                    ocr_text = ocr_result[0][0][1][0]
                    ocr_confidence = ocr_result[0][0][1][1]
                else:
                    ocr_text = "؟"
                    ocr_confidence = 0.0

                # Add RLO character to ensure right-to-left direction for Arabic
                if detected_language == 'ar':
                    ocr_text = '\u202E' + ocr_text[::-1]

                # Compare STR and PaddleOCR confidence scores and select the best result
                if confidence_score_str >= ocr_confidence:
                    best_text = pred
                    best_confidence = confidence_score_str
                else:
                    best_text = ocr_text
                    best_confidence = ocr_confidence

                print(f'{img_name:25s}\t{best_text:25s}\t{best_confidence:0.4f}')
                log.write(f'{img_name:25s}\t{best_text:25s}\t{best_confidence:0.4f}\n')

                llm = createModel()
                text = best_text
                trg_lang = 'fr'
                src_lang = detected_language

                res = generateTrans(llm, text, trg_lang, src_lang)

                #print(res)

                txt2spch(res, trg_lang)

                # Write the best result to the result file
                with open(f'./ocr_results.txt', 'a', encoding='utf-8') as result_file:
                    result_file.write(f'{img_name:25s}\t{best_text:25s}\t{res}\n')

            log.close()

if __name__ == '__main__':
    complete_characters = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "أبجدهوزيكلمنسعفصقرشتثخحطضظغؤئذى"
        "的一是不了我人在他有这中大来上个国为以和出地道于时要就下生会自面看也年得后多小没还之过天去好让你听知回事自己现如一样可子无做各爱于小啊都想能那该谁再我们什么啊吧嗯哦哎这样对好吧看来来吧哦天我的神也许继续没事没问题完成人们相信"
        "가나다라마바사아자차카타파하늘바람과별과시내에는너를랑해"
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも"
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモ"
        "অআইঈউঊএঐওকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
        "अआइईउऊऋऌएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        "،؟"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=complete_characters, help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=64, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
