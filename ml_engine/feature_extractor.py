import cv2
import numpy as np
import pywt
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from scipy.stats import skew

logger = logging.getLogger('ml_engine')


class ImageFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.deep_model = nn.Sequential(*list(base_model.children())[:-1])
        self.deep_model.to(self.device)
        self.deep_model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_input, session=None):
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
            else:
                img = image_input
            if img is None: return None
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            img_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
            img_denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, 10, 10, 7, 21)
            try:
                result_rgba = remove(img_denoised, session=session) if session else remove(img_denoised)
            except:
                result_rgba = remove(img_denoised)
            if result_rgba.shape[2] == 4:
                alpha = result_rgba[:, :, 3]
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
                if np.sum(mask) == 0: return None
                segmented_img = cv2.bitwise_or(img_denoised, img_denoised, mask=mask)
                gray_segmented = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
                return segmented_img, gray_segmented, mask
        except:
            return None

    def _get_manual_features(self, segmented_img, gray, mask):
        vec = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            vec.append(cv2.arcLength(cnt, True))
            _, radius = cv2.minEnclosingCircle(cnt)
            vec.append(radius * 2)
            M = cv2.moments(cnt)
            vec.extend([M['m10'] / M['m00'], M['m01'] / M['m00']] if M['m00'] != 0 else [0, 0])  # COG

            complex_b = cnt.reshape(-1, 2)[:, 0] + 1j * cnt.reshape(-1, 2)[:, 1]
            fft = np.abs(np.fft.fft(complex_b))
            fft = (fft / fft[0])[1:5].tolist() if fft[0] != 0 else [0, 0, 0, 0]
            vec.extend(fft + [0.0] * (4 - len(fft)))

            contours_cc, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_cc = max(contours_cc, key=cv2.contourArea)
            cc, lk = [], {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
            for i in range(len(cnt_cc) - 1):
                move = tuple(np.sign(cnt_cc[i + 1][0] - cnt_cc[i][0]))
                if move in lk: cc.append(lk[move])
            vec.extend([np.mean(cc), np.mean([(cc[i] - cc[i - 1]) % 8 for i in range(len(cc))])] if cc else [0, 0])

            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            angles = []
            for i in range(len(approx)):
                p1, p2, p3 = approx[i - 1][0], approx[i][0], approx[(i + 1) % len(approx)][0]
                v1, v2 = p1 - p2, p3 - p2
                angles.append(
                    np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10), -1.0, 1.0)))
            vec.append(np.mean(angles) if angles else 0)
        else:
            vec.extend([0] * 11)

        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a, p = cv2.contourArea(c), cv2.arcLength(c, True)
            vec.append(a)
            vec.append((4 * np.pi * a) / (p ** 2) if p > 0 else 0)
            vec.append((p ** 2) / a if a > 0 else 0)
            _, (w, h), _ = cv2.minAreaRect(c)
            vec.append(a / (w * h) if w * h > 0 else 0)
            if len(c) >= 5:
                _, (MA, ma), _ = cv2.fitEllipse(c)
                vec.append(np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) > 0 else 0)
            else:
                vec.append(0)
            if hier is not None:
                objs = sum(1 for i in range(len(cnts)) if hier[0][i][3] == -1)
                hols = sum(1 for i in range(len(cnts)) if hier[0][i][3] != -1)
                vec.append(objs - hols)  # Euler
                t_obj_a = sum(cv2.contourArea(ci) for i, ci in enumerate(cnts) if hier[0][i][3] == -1)
                t_hol_a = sum(cv2.contourArea(ci) for i, ci in enumerate(cnts) if hier[0][i][3] != -1)
                vec.append(t_hol_a / t_obj_a if t_obj_a > 0 else 0)
            else:
                vec.extend([1, 0])
            vec.append(cv2.contourArea(cv2.convexHull(c)))
        else:
            vec.extend([0] * 8)

        glcm = graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, True, True)
        for pr in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            vec.append(graycoprops(glcm, pr).mean())
        vec.append(np.mean(
            [-np.sum(glcm[:, :, 0, i][glcm[:, :, 0, i] > 0] * np.log2(glcm[:, :, 0, i][glcm[:, :, 0, i] > 0] + 1e-10))
             for i in range(4)]))

        f_s = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.abs(f_s)
        vec.extend([np.mean(mag), np.max(mag)])  # Fourier Spectrum
        vec.extend(cv2.dct(np.float32(cv2.resize(gray, (64, 64))) / 255.0)[0:5, 0:5].flatten()[1:].tolist())  # DCT
        _, (lh, hl, hh) = pywt.dwt2(gray, 'haar')  # Wavelet
        for b in [lh, hl, hh]: vec.extend([np.mean(b), np.mean(np.square(b))])

        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)  # Harris
        vec.extend([np.mean(dst), np.sum(dst > 0.01 * dst.max())])

        try:
            sift = cv2.SIFT_create(nfeatures=128)
            _, des = sift.detectAndCompute(gray, None)
            vec.extend(np.mean(des, axis=0).tolist() if des is not None else [0.0] * 128)
        except:
            vec.extend([0.0] * 128)

        lab = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2LAB)
        m, s = cv2.meanStdDev(lab, mask=mask)
        vec.extend(m.flatten().tolist() + s.flatten().tolist())

        hsv = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)
        vec.extend(cv2.normalize(cv2.calcHist([hsv], [0], mask, [8], [0, 180]), None).flatten().tolist())  # H Hist
        vec.extend(cv2.normalize(cv2.calcHist([hsv], [1], mask, [8], [0, 256]), None).flatten().tolist())  # S Hist
        for i in range(3):
            ch = hsv[:, :, i][mask > 0]
            vec.extend([np.mean(ch), np.std(ch), skew(ch) if len(ch) > 0 else 0])

        lbp = local_binary_pattern(gray, 8, 1, 'uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        vec.extend((hist_lbp.astype(float) / (hist_lbp.sum() + 1e-7)).tolist())

        vec.extend(hog(cv2.resize(gray, (64, 64)), 9, (8, 8), (2, 2)).tolist())  # HOG

        return np.nan_to_num(vec).tolist()

    def extract_features_from_processed(self, s_img, g_img, mask):
        m_vec = self._get_manual_features(s_img, g_img, mask)
        pil_img = Image.fromarray(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))
        inp = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d_vec = self.deep_model(inp).cpu().flatten().numpy().tolist()
        return m_vec + d_vec

    def extract_all_features(self, path, session=None):
        r = self.preprocess_image(path, session)
        if not r: return None
        return self.extract_features_from_processed(r[0], r[1], r[2])