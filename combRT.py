import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as compare_ssim
from collections import OrderedDict
import time
import argparse
import arabic_reshaper
from bidi.algorithm import get_display
from imutils.video import VideoStream, FPS


# Your existing imports for CRAFT and recognition model
from utils import CTCLabelConverter, AttnLabelConverter
from datasetvid import RawDataset, AlignCollate
from model import Model
import craft_utils
import imgproc
import file_utils
from craft import CRAFT

from langdetect import detect, LangDetectException
from llama import createModel, generateTrans
from ttsg import txt2spch
from paddleocr import PaddleOCR


# Initialize Llama model for translation
llm = createModel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FONT_PATHS = {
    'en': 'fontxt/noto-sans.regular.ttf',        # English font
    'ar': 'fontxt/NotoKufiArabic-Regular.ttf',  # Arabic font
    'zh': 'fontxt/NotoSansTC-Regular.ttf',      # Simplified Chinese font
    'ko': 'fontxt/NotoSansKR-Regular.ttf',      # Korean font
    'ja': 'fontxt/NotoSansJP-Regular.ttf',      # Japanese font
    'hi': 'fontxt/NotoSansDevanagari-Regular.ttf',  # Hindi/Devanagari script font
    'bn': 'fontxt/NotoSansBengali-Regular.ttf',     # Bengali font
    'fr': 'fontxt/noto-sans.regular.ttf',            
    'de': 'fontxt/noto-sans.regular.ttf',           
    'it': 'fontxt/noto-sans.regular.ttf'  
}

def draw_text_with_pil(frame, text, position, language, font_size=20, color=(255, 0, 0)):
    font_path = FONT_PATHS.get(language, FONT_PATHS['en'])  # Default to English if no font found
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)

    # Adjust text for Arabic (or other RTL language)
    if language == 'ar':
        reshaped_text = arabic_reshaper.reshape(text)  # Reshape Arabic letters
        text = get_display(reshaped_text)         # Apply bidi algorithm to reorder text

    # Draw text over the frame
    draw.text(position, text, font=font, fill=color)

    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

def is_key_frame(prev_frame, current_frame, threshold=0.8):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    ssim, _ = compare_ssim(prev_gray, curr_gray, full=True)

def is_new_text(prev_texts, current_texts):
    # Extract only the first element (text) from each entry
    current_text_set = {entry[0] for entry in current_texts if len(entry) > 1}
    prev_text_set = {entry[0] for entry in prev_texts if len(entry) > 1}
    return current_text_set != prev_text_set

def enhance_image_quality(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def load_text_recognition_model(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 1

    model = Model(opt)
    model = model.to(device)

    checkpoint = torch.load(opt.saved_model, map_location=device)

    state_dict = checkpoint
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model, converter

def recognize_text(image, model, converter, opt):
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    image_tensors = AlignCollate_demo([image])
    batch_size = image_tensors.size(0)
    image = image_tensors.to(device)

    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    with torch.no_grad():
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
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
        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]
            pred_max_prob = pred_max_prob[:pred_EOS]

        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

    return pred, confidence_score

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, opt.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=opt.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys, score_text

def combine_boxes_and_translate(detected_texts):
    # If no detected texts, return an empty list immediately
    if not detected_texts:
        return []

    # Sort detected texts by y-coordinate (top to bottom), then by x-coordinate (left to right)
    sorted_texts = sorted(detected_texts, key=lambda x: (x[1][1], x[1][0]))
    phrases = []
    current_phrase = sorted_texts[0][0]
    current_box = sorted_texts[0][1]

    for i in range(1, len(sorted_texts)):
        text, (x_min, y_min, x_max, y_max) = sorted_texts[i]
        prev_x_min, prev_y_min, prev_x_max, prev_y_max = current_box
        
        # If boxes are close, combine text into a single phrase
        if abs(x_min - prev_x_max) < 10 or abs(y_min - prev_y_max) < 10:
            current_phrase += " " + text
            current_box = (min(x_min, prev_x_min), min(y_min, prev_y_min), max(x_max, prev_x_max), max(y_max, prev_y_max))
        else:
            phrases.append((current_phrase, current_box))
            current_phrase, current_box = text, (x_min, y_min, x_max, y_max)
    
    # Append the last phrase to the list
    phrases.append((current_phrase, current_box))
    return phrases

def process_frame(frame, net, text_recognition_model, converter, opt, target_language):
    # 1. Text detection
    boxes, polys, _ = test_net(net, frame, opt.text_threshold, opt.link_threshold, opt.low_text, opt.cuda, opt.poly)
    detected_texts = []
    total_confidence = 0

    for box in boxes:
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)

        # Draw the bounding box around the detected text before recognition (CRAFT output)
        frame = cv2.polylines(frame, [np.int32(box)], isClosed=True, color=(0, 255, 0), thickness=1)

        # Crop and check if the image is empty
        cropped_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        if cropped_image.size == 0:
            continue  # Skip to the next box if the crop is empty

        # Apply enhancement only if cropped_image is not empty
        cropped_image = enhance_image_quality(cropped_image)

        # Initialize orig_text and orig_confidence to default values
        orig_text, orig_confidence = "", 0.0

        # 2. Recognition with STR and OCR
        if cropped_image.size > 0:  # Ensure the image is valid
            orig_text, orig_confidence = recognize_text(cropped_image, text_recognition_model, converter, opt)
            try:
                detected_language = detect(orig_text)
            except LangDetectException:
                detected_language = 'en'

            ocr = get_paddleocr_instance(detected_language)
            ocr_results = ocr.ocr(np.array(cropped_image), cls=True)
            paddle_text, paddle_confidence = "", 0.0

            if ocr_results and ocr_results[0]:
                paddle_text, paddle_confidence = ocr_results[0][0][1][0], ocr_results[0][0][1][1]
                if detected_language == 'ar':
                    paddle_text = '\u202E' + paddle_text[::-1]

            # 3. Choose the best result and add to detected_texts
            best_text, best_confidence = (paddle_text, paddle_confidence) if paddle_confidence > orig_confidence else (orig_text, orig_confidence)
            if best_confidence >= 0.5:
                detected_texts.append((best_text, best_confidence, (x_min, y_min, x_max, y_max)))
                total_confidence += best_confidence
                frame = draw_text_with_pil(frame, best_text, (int(x_min), int(y_min) - 30), detected_language, color=(255, 0, 0))


    # 4. Group and translate text
    translated_text = ""  # Default to an empty string if no translation occurs
    phrases = combine_boxes_and_translate([(text, box) for text, _, box in detected_texts])
    for phrase, box in phrases:
        translated_text = generateTrans(llm, phrase, target_language, detected_language)
        txt2spch(translated_text, target_language)
        x_min, y_min, x_max, y_max = box
        frame = draw_text_with_pil(frame, translated_text, (int(x_min), int(y_min) - 50), detected_language, color=(0, 0, 255))


    # Write results to a text file if there's detected text
    if detected_texts:
        with open("rt.txt", "a", encoding="utf-8") as text_file:
            text_file.write(f"Frame {frame_count} - STR Text: {orig_text} (Confidence: {orig_confidence:.2f}), PaddleOCR Text: {paddle_text} (Confidence: {paddle_confidence:.2f}), Translation: {translated_text}\n")

    avg_confidence = total_confidence / len(detected_texts) if detected_texts else 0
    return frame, avg_confidence, detected_texts


if __name__ == '__main__':

    supported_languages = {
    'English': 'en',
    'Arabic': 'ar',
    'Chinese (Mandarin)': 'zh',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Hindi': 'hi',
    'Dutch': 'nl',
    'Turkish': 'tr',
    'Swedish': 'sv',
    'Danish': 'da',
    'Polish': 'pl',
    'Finnish': 'fi',
    'Czech': 'cs',
    'Greek': 'el',
    'Romanian': 'ro',
    'Hungarian': 'hu',
    'Hebrew': 'he',
    'Bengali': 'bn',
    'Indonesian': 'id',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Malay': 'ms',
    'Filipino': 'tl',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Urdu': 'ur',
    'Swahili': 'sw'
}


    for language, code in supported_languages.items():
        print(f"{language}: {code}")

    target_language = input("Enter your language (exp : fr for French) : ")

    txt2spch("Hello! I am your voice assistant, here is the text written in the direction of your camera translated into your language", 'en')

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
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--video', type=str, help='path to video file')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    #parser.add_argument('--sensitive', action='store_true', help='Use case-sensitive mode')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=complete_characters, help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=64, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    text_recognition_model, converter = load_text_recognition_model(opt)

    # Initialiser le modèle CRAFT
    net = CRAFT()
    if opt.cuda:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cuda')))
        net.cuda()
    else:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cpu')))
    net.eval()

    if opt.refine:
        refine_net = CRAFT()
        if opt.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model, map_location='cuda')))
            refine_net.cuda()
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model, map_location='cpu')))
        refine_net.eval()
    else:
        refine_net = None

    # Initialiser PaddleOCR pour le post-traitement
    #paddle_ocr = PaddleOCR(lang='multilingual')  # Langue définie pour l'OCR
    '''
    camera_url = "rtsp://http://10.0.51.190:4747/video"

    # Open video stream
    cap = cv2.VideoCapture(camera_url)
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        exit()

    frame_count = 0
    last_processed_frame = 0
    prev_detected_texts = []
    processed_frames = []
    prev_frame = None

    # Intervalle pour forcer la détection et la traduction (en frames)
    refresh_interval = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la caméra.")
            break
        
        frame_count += 1
        
        # Traiter le cadre si c'est un frame clé ou si l'intervalle de rafraîchissement est atteint
        if prev_frame is None or frame_count - last_processed_frame >= refresh_interval:
            last_processed_frame = frame_count
            prev_frame = frame

            # Traitement du cadre et reconnaissance de texte
            processed_frame, avg_confidence, detected_texts = process_frame(
                frame, net, text_recognition_model, converter, opt, target_language
            )

            # Vérifier si le texte détecté est nouveau
            if is_new_text(prev_detected_texts, detected_texts):
                # Sauvegarder les meilleurs résultats (frame, texte, confiance)
                best_frame, best_score, best_texts = max(
                    processed_frames, key=lambda x: x[1], default=(None, 0, [])
                )

                # Sauvegarde des résultats si texte nouveau détecté
                if best_frame is not None:
                    print(f"Texte détecté: {best_texts}")

                # Redémarrer le stockage des frames traités
                processed_frames = []

            # Ajouter le cadre actuel à la séquence
            processed_frames.append((processed_frame, avg_confidence, detected_texts))
            prev_detected_texts = detected_texts

            # Afficher le cadre (facultatif pour le debugging)
            cv2.imshow('Video', processed_frame)

        # Gestion de sortie
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''

    #cam_url = "http://10.0.51.190:4747/video"

    # Open video stream

    if not opt.video == False:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1)

    else:
        vs = cv2.VideoCapture(opt['video'])

    fps = FPS().start()

    frame_count = 0
    last_processed_frame = 0
    prev_detected_texts = []
    processed_frames = []
    prev_frame = None

    # Intervalle pour forcer la détection et la traduction (en frames)
    refresh_interval = 30

    while True:

        frame = vs.read()
        frame = frame[1] if opt.video == False else frame

        if frame is None:
            break

        frame_count += 1
        
        # Traiter le cadre si c'est un frame clé ou si l'intervalle de rafraîchissement est atteint
        if prev_frame is None or frame_count - last_processed_frame >= refresh_interval:
            last_processed_frame = frame_count
            prev_frame = frame

            # Traitement du cadre et reconnaissance de texte
            processed_frame, avg_confidence, detected_texts = process_frame(
                frame, net, text_recognition_model, converter, opt, target_language
            )

            # Vérifier si le texte détecté est nouveau
            if is_new_text(prev_detected_texts, detected_texts):
                # Sauvegarder les meilleurs résultats (frame, texte, confiance)
                best_frame, best_score, best_texts = max(
                    processed_frames, key=lambda x: x[1], default=(None, 0, [])
                )

                # Sauvegarde des résultats si texte nouveau détecté
                if best_frame is not None:
                    print(f"Texte détecté: {best_texts}")

                # Redémarrer le stockage des frames traités
                processed_frames = []

            # Ajouter le cadre actuel à la séquence
            processed_frames.append((processed_frame, avg_confidence, detected_texts))
            prev_detected_texts = detected_texts

        fps.update()

        # Afficher le cadre (facultatif pour le debugging)
        cv2.imshow('Video', processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    fps.stop()
    print(f"[INFO] elapsed time {round(fps.elapsed(), 2)}")
    print(f"[INFO] approx. FPS : {round(fps.fps(), 2)}")

    if not opt.video == False:
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()
