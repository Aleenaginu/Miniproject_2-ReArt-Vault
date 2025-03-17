import json
import base64
import numpy as np
import cv2
from django.http import JsonResponse
from django.contrib.auth import get_user_model, login, authenticate
from django.views.decorators.csrf import csrf_exempt
from .models import FaceEncoding
import logging

logger = logging.getLogger(__name__)
User = get_user_model()

def base64_to_image(base64_string):
    try:
        # Remove the data URL prefix if present
        if 'data:image' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        logger.error(f"Error in base64_to_image: {str(e)}")
        raise

def get_face_encoding(image, face):
    """Extract face encoding from image with improved alignment and normalization"""
    try:
        x, y, w, h = face
        # Add adaptive padding based on face size
        padding_percent = min(0.3, max(0.1, w / image.shape[1]))  # Adaptive padding 10-30%
        padding = int(w * padding_percent)
        
        # Add padding with bounds checking
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        face_roi = image[y:y+h, x:x+w]
        
        # Convert to grayscale for better feature extraction
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_roi = clahe.apply(face_roi)
        
        # Resize maintaining aspect ratio
        target_size = (128, 128)
        aspect = w / h
        if aspect > 1:
            new_w = target_size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect)
        
        face_roi = cv2.resize(face_roi, (new_w, new_h))
        
        # Pad to square if needed
        if new_w != target_size[0] or new_h != target_size[1]:
            delta_w = target_size[0] - new_w
            delta_h = target_size[1] - new_h
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            face_roi = cv2.copyMakeBorder(face_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # Get face encoding
        face_encoding = face_roi.flatten()
        
        # Robust normalization
        face_encoding = (face_encoding - np.mean(face_encoding)) / (np.std(face_encoding) + 1e-7)
        return face_encoding
    except Exception as e:
        logger.error(f"Error in get_face_encoding: {str(e)}")
        raise

@csrf_exempt
def store_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            image_data = data.get('image')
            
            if not all([username, password, image_data]):
                return JsonResponse({'error': 'Missing required fields'}, status=400)
            
            # Verify credentials
            user = authenticate(username=username, password=password)
            if not user:
                return JsonResponse({'error': 'Invalid credentials'}, status=401)
            
            # Process image
            image = base64_to_image(image_data)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face with improved parameters
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # More precise scaling
                minNeighbors=6,   # More strict neighbor requirement
                minSize=(100, 100),
                maxSize=(500, 500),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return JsonResponse({
                    'error': 'No face detected. Please ensure:\n' +
                            '1. Your face is well-lit\n' +
                            '2. You are facing the camera directly\n' +
                            '3. There are no strong shadows on your face'
                }, status=400)
            
            if len(faces) > 1:
                return JsonResponse({
                    'error': 'Multiple faces detected. Please ensure:\n' +
                            '1. Only your face is visible in the camera\n' +
                            '2. There are no faces in the background\n' +
                            '3. There are no face-like objects in view'
                }, status=400)
            
            # Get face encoding with improved processing
            face_encoding = get_face_encoding(image, faces[0])
            
            # Store encoding
            try:
                face_record, created = FaceEncoding.objects.get_or_create(user=user)
                face_record.encoding = json.dumps(face_encoding.tolist())
                face_record.save()
                
                return JsonResponse({'message': 'Face enrolled successfully'})
            except Exception as e:
                logger.error(f"Error storing face encoding: {str(e)}")
                return JsonResponse({'error': 'Failed to store face encoding'}, status=500)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error in store_face: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
