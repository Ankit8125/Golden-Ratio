import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

GOLDEN_RATIO = 1.618

def calculate_distance(point1, point2):
    if point1 is None or point2 is None:
        return 0
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_golden_ratio_score(face_landmarks, image_width, image_height):
    def get_pixel_coords(landmark_id):
        lm = face_landmarks.landmark[landmark_id]
        return mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, image_width, image_height)

    essential_landmarks = [
        10, 152, 234, 454,
        1, 2, 101, 330,
        13, 14, 33, 263,
        164, 397, 159, 145,
        386, 374, 78, 308,
        11, 12, 168
    ]

    for idx in essential_landmarks:
        if get_pixel_coords(idx) is None:
            return 0.0

    forehead_point = get_pixel_coords(10)
    chin_point = get_pixel_coords(152)
    left_cheek_point = get_pixel_coords(234)
    right_cheek_point = get_pixel_coords(454)
    face_length = calculate_distance(forehead_point, chin_point)
    face_width = calculate_distance(left_cheek_point, right_cheek_point)
    face_ratio = face_length / face_width if face_width != 0 else 0

    nose_top = get_pixel_coords(1)
    nose_base = get_pixel_coords(2)
    left_nostril = get_pixel_coords(101)
    right_nostril = get_pixel_coords(330)
    nose_length = calculate_distance(nose_top, nose_base)
    nose_width = calculate_distance(left_nostril, right_nostril)
    nose_ratio = nose_length / nose_width if nose_width != 0 else 0

    left_eye_center = get_pixel_coords(11)
    right_eye_center = get_pixel_coords(12)
    if left_eye_center and right_eye_center:
        eyes_midpoint = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
    else:
        eyes_midpoint = None

    mouth_left = get_pixel_coords(78)
    mouth_right = get_pixel_coords(308)
    if mouth_left and mouth_right:
        mouth_midpoint = ((mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2)
    else:
        mouth_midpoint = None

    eye_to_mouth_distance = calculate_distance(eyes_midpoint, mouth_midpoint)
    eye_mouth_nose_ratio = eye_to_mouth_distance / nose_length if nose_length != 0 else 0

    interocular_distance = calculate_distance(left_eye_center, right_eye_center)
    interocular_nose_ratio = interocular_distance / nose_width if nose_width != 0 else 0

    lip_width = calculate_distance(mouth_left, mouth_right)
    lip_nose_ratio = lip_width / nose_width if nose_width != 0 else 0

    def score_ratio(ratio):
        if ratio == 0:
            return 0
        return 1 - abs(ratio - GOLDEN_RATIO) / GOLDEN_RATIO

    score_face = score_ratio(face_ratio)
    score_nose = score_ratio(nose_ratio)
    score_eye_mouth_nose = score_ratio(eye_mouth_nose_ratio)
    score_interocular_nose = score_ratio(interocular_nose_ratio)
    score_lip_nose = score_ratio(lip_nose_ratio)

    valid_scores = [s for s in [score_face, score_nose, score_eye_mouth_nose, score_interocular_nose, score_lip_nose] if s > 0]
    if not valid_scores:
        return 0.0

    overall_score = sum(valid_scores) / len(valid_scores)
    return max(0, min(1, overall_score)) * 100

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,  # Allow up to 5 faces
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

                score = calculate_golden_ratio_score(face_landmarks, image_width, image_height)

                forehead = face_landmarks.landmark[10]
                forehead_coords = mp_drawing._normalized_to_pixel_coordinates(forehead.x, forehead.y, image_width, image_height)
                if forehead_coords:
                    cv2.putText(image, f"Score: {score:.2f}%", (forehead_coords[0], forehead_coords[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Golden Ratio (Multi-Face)", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
