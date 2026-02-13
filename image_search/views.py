from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.apps import apps
import numpy as np
import cv2
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('cbir_app')


def index(request):
    context = {}
    config = apps.get_app_config('image_search')
    artifacts = config.ml_artifacts
    fs = FileSystemStorage()

    if request.method == 'POST' and request.FILES.get('myfile'):
        try:
            img_file = request.FILES['myfile']
            filename = fs.save(img_file.name, img_file)
            original_url = fs.url(filename)
            original_path = fs.path(filename)

            # 1. Run Preprocessing once to get images and mask
            res = config.extractor.preprocess_image(original_path, session=config.rembg_session)

            if res:
                segmented_img, gray_segmented, mask = res

                # Save segmented image for UI display
                _, enc = cv2.imencode('.jpg', segmented_img)
                p_filename = fs.save(f"proc_{img_file.name}", ContentFile(enc.tobytes()))
                p_url = fs.url(p_filename)

                # 2. Extract features using the already processed data (No redundant rembg call)
                features = config.extractor.extract_features_from_processed(segmented_img, gray_segmented, mask)

                if features:
                    f_array = np.array(features).reshape(1, -1)
                    f_scaled = artifacts['scaler'].transform(f_array)
                    f_pca = artifacts['pca'].transform(f_scaled)

                    # Store for Relevance Feedback
                    request.session['current_query_vector'] = f_pca.tolist()

                    return perform_search(request, f_pca, artifacts, original_url, p_url, context)
            else:
                context['error'] = "Preprocessing failed. Try a clearer image."

        except Exception as e:
            logger.error(f"Upload Error: {e}")
            context['error'] = "An error occurred during image processing."

    elif request.method == 'POST' and 'refine' in request.POST:
        selected_urls = request.POST.getlist('selected_images')
        original_query_vector = request.session.get('current_query_vector')

        if selected_urls and original_query_vector:
            original_vector = np.array(original_query_vector)
            db_paths = artifacts['db_paths']
            db_features = artifacts['db_features']

            selected_vectors = []
            for url in selected_urls:
                # Cleaning URL to match database paths
                rel_path = url.split('/media/')[-1].replace('/', '\\')
                for i, path in enumerate(db_paths):
                    if rel_path in path:
                        selected_vectors.append(db_features[i])
                        break

            if selected_vectors:
                # Relevance Feedback: 30% Original Query, 70% User Selection
                mean_selected = np.mean(selected_vectors, axis=0).reshape(1, -1)
                refined_vector = (0.3 * original_vector) + (0.7 * mean_selected)

                request.session['current_query_vector'] = refined_vector.tolist()

                return perform_search(request, refined_vector, artifacts,
                                      request.POST.get('original_query_url'),
                                      request.POST.get('processed_url'), context)

    return render(request, 'index.html', context)


def perform_search(request, query_vector, artifacts, uploaded_url, processed_url, context):
    # Predict class to limit search scope
    class_idx = artifacts['model'].predict(query_vector)[0]
    predicted_label = artifacts['label_encoder'].inverse_transform([class_idx])[0]

    db_labels = artifacts['db_labels']
    mask = (db_labels == predicted_label)
    target_features = artifacts['db_features'][mask]
    target_paths = artifacts['db_paths'][mask]

    if len(target_features) > 0:
        # Using Cosine Similarity for high-dimensional feature vectors
        scores = cosine_similarity(query_vector, target_features)[0]

        top_k = min(10, len(target_features))
        indices = np.argsort(scores)[::-1][:top_k]

        similar_images = []
        for idx in indices:
            rel = target_paths[idx].split('media')[-1].replace('\\', '/')
            similar_images.append({
                'url': f"/media/{rel.lstrip('/')}",
                'score': f"{scores[idx]:.4f}"
            })

        context.update({
            'uploaded_url': uploaded_url,
            'processed_url': processed_url,
            'predicted_class': predicted_label,
            'similar_images': similar_images
        })
    else:
        context['error'] = f"No images found for the predicted category: {predicted_label}"

    return render(request, 'index.html', context)