import os, glob, logging, joblib, warnings, numpy as np
from tqdm import tqdm
from collections import Counter
from rembg import new_session
import onnxruntime as ort
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from feature_extractor import ImageFeatureExtractor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('trainer')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, 'media')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'artifacts')
if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)


def get_label(p):
    parts = p.lower().replace('\\', '/').split('/')
    cls_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    for c in cls_list:
        if c in parts: return 'aeroplane' if c == 'airplane' else ('cow' if c == 'cows' else c)
    return parts[-2]


def train_system():
    files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.[jJ][pP][gG]"), recursive=True)
    logger.info(f"Total images found in {DATASET_ROOT}: {len(files)}")

    if len(files) == 0:
        logger.error("No images found! Ensure 'media' folder contains JPG files.")
        return

    temp_f, temp_l = [], []
    for f in files:
        l = get_label(f)
        if l: temp_f.append(f); temp_l.append(l)

    counts = Counter(temp_l)
    valid_data = [(f, l) for f, l in zip(temp_f, temp_l) if counts[l] >= 2]

    if not valid_data:
        logger.error("No valid data after class filtering. Check folder structure.")
        return

    valid_f, valid_l = zip(*valid_data)
    logger.info(f"Processing {len(valid_f)} images across {len(set(valid_l))} classes...")

    ext = ImageFeatureExtractor()
    sess = new_session("u2net", providers=['CUDAExecutionProvider',
                                           'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else [
        'CPUExecutionProvider'])

    X_list, y_list, path_list = [], [], []
    for p in tqdm(valid_f, desc="Hybrid Extraction"):
        feats = ext.extract_all_features(p, sess)
        if feats:
            X_list.append(feats)
            current_label = valid_l[valid_f.index(p)]
            y_list.append(current_label)
            path_list.append(p)

    if not X_list:
        logger.error("Feature extraction failed for all images. Check Preprocessing.")
        return

    X = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_list)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    sc_val = StandardScaler().fit(X_train)
    pca_val = PCA(n_components=0.95, random_state=42).fit(sc_val.transform(X_train))
    X_train_pca = pca_val.transform(sc_val.transform(X_train))

    logger.info("Tuning Models...")
    xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss'), {
        'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]
    }, n_iter=4, cv=3, n_jobs=-1).fit(X_train_pca, y_train)

    lgbm_search = RandomizedSearchCV(LGBMClassifier(verbose=-1), {
        'n_estimators': [100, 300], 'num_leaves': [31, 63]
    }, n_iter=4, cv=3, n_jobs=-1).fit(X_train_pca, y_train)

    logger.info("Final training on 100% data...")
    final_sc = StandardScaler().fit(X)
    final_pca = PCA(n_components=0.95, random_state=42).fit(final_sc.transform(X))
    X_all_pca = final_pca.transform(final_sc.transform(X))

    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)),
            ('lgbm', lgbm_search.best_estimator_),
            ('xgb', xgb_search.best_estimator_)
        ],
        final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1
    ).fit(X_all_pca, y)

    joblib.dump({
        'scaler': final_sc,
        'pca': final_pca,
        'model': stack,
        'label_encoder': le,
        'db_features': X_all_pca,
        'db_paths': np.array(path_list),
        'db_labels': le.inverse_transform(y)
    }, os.path.join(ARTIFACTS_DIR, 'system_core.pkl'))

    logger.info(f"Training success! Final model saved with {X.shape[1]} original features.")


if __name__ == "__main__":
    train_system()