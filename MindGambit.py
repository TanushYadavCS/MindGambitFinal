import mediapipe as mp
import cv2 as cv
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
moveArr = ["Rock", "Paper", "Scissors"]


def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    # Calculate the average y-coordinate for the fingertips
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Determine the gesture based on finger tip positions
    if (
        thumb_tip_y < index_tip_y
        and thumb_tip_y < middle_tip_y
        and thumb_tip_y < ring_tip_y
        and thumb_tip_y < pinky_tip_y
    ):
        return "rock"
    elif (
        index_tip_y < thumb_tip_y
        and middle_tip_y < thumb_tip_y
        and ring_tip_y < thumb_tip_y
        and pinky_tip_y < thumb_tip_y
    ):
        return "paper"
    else:
        return "scissors"


cap = cv.VideoCapture(0)
clock = 0
score = 0
gameText = ""
player_move = None
success = True

cv.namedWindow("frame", cv.WINDOW_NORMAL)

player_detected = False

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                player_detected = True
            else:
                player_detected = False

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        frame = cv.flip(frame, 1)
        if 0 <= clock < 20:
            success = True
            gameText = "Ready..?"
        elif clock < 30:
            gameText = "3.."
        elif clock < 40:
            gameText = "2.."
        elif clock < 50:
            gameText = "1.."
        elif clock < 60:
            gameText = "GO!"
        elif clock == 60:
            hls = results.multi_hand_landmarks
            if player_detected:
                player_move = getHandMove(hls[0])
            else:
                success = False
        elif clock < 100:
            if success:
                if player_move == "rock":
                    gameText = "Paper"
                elif player_move == "paper":
                    gameText = "Scissors"
                elif player_move == "scissors":
                    gameText = "Rock"
            else:
                gameText = "Didn't play properly!"
        cv.putText(
            frame,
            gameText,
            (273, 262),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 197),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            f"Clock: {clock}",
            (50, 50),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (243, 255, 0),
            2,
            cv.LINE_AA,
        )

        clock = (clock + 1) % 100
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
