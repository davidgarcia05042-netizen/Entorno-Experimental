"""
Diagnóstico: revisa si el umbral de confianza de MediaPipe
(min_detection_confidence, default 0.5 en MediaPipePoseEstimator) está
bien configurado, o si la correlación débil vista en 3EJ7 (pie derecho,
r=0.701 en el mejor caso, ni siquiera en tobillo/pie) se explica por baja
visibilidad de esa zona (ej. sombra).

Reporta, para cada video de prueba ya usado en _diag_marker_alignment.py:
visibilidad media/mínima de tobillo y pie (izq/der), y qué fracción de
frames cae por debajo de distintos umbrales de confianza.
"""

import cv2
import mediapipe as mp

VIDEOS = [
    ("2EJ1 - brazos", "data/gold_standard/TOMA FRONTAL/2EJ1/2EJ1.mp4"),
    ("2EJ3 - antebrazo", "data/gold_standard/TOMA FRONTAL/2EJ3/2EJ3.mp4"),
    ("2EJ5 - pierna derecha", "data/gold_standard/TOMA FRONTAL/2EJ5/2EJ5.mp4"),
    ("3EJ7 - pie derecho", "data/gold_standard/TOMA FRONTAL/3EJ7/3EJ7.mp4"),
    ("3EJ10 - sentadilla", "data/gold_standard/TOMA FRONTAL/3EJ10/3EJ10.mp4"),
]

# índices MediaPipe relevantes para pie/tobillo
LANDMARKS_OF_INTEREST = {
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

THRESHOLDS = [0.9, 0.7, 0.5, 0.3]


def main() -> None:
    mp_pose = mp.solutions.pose

    for label, video_path in VIDEOS:
        cap = cv2.VideoCapture(video_path)
        visibilities = {name: [] for name in LANDMARKS_OF_INTEREST}
        person_detected = 0
        total = 0
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total += 1
                results = pose.process(frame[:, :, ::-1])
                if results.pose_landmarks is None:
                    continue
                person_detected += 1
                lm = results.pose_landmarks.landmark
                for name, idx in LANDMARKS_OF_INTEREST.items():
                    visibilities[name].append(lm[idx].visibility)
        cap.release()

        print("=" * 70)
        print(f"{label} -- persona detectada en {person_detected}/{total} frames ({100*person_detected/total:.1f}%)")
        print(f"{'landmark':18s} {'media':>7s} {'min':>7s} " + " ".join(f'<{t:.1f}' for t in THRESHOLDS))
        for name, vis in visibilities.items():
            if not vis:
                print(f"{name:18s}  (sin datos)")
                continue
            mean_v = sum(vis) / len(vis)
            min_v = min(vis)
            fracs = [f"{100*sum(1 for v in vis if v < t)/len(vis):5.1f}%" for t in THRESHOLDS]
            print(f"{name:18s} {mean_v:7.3f} {min_v:7.3f} " + " ".join(fracs))
        print()


if __name__ == "__main__":
    main()
