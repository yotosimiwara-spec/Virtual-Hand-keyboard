import cv2
import mediapipe as mp
import time
class Button():
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img, hover=False):
        x, y = self.pos
        w, h = self.size
        color = (0, 255, 0) if hover else (69, 63, 61)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(img, self.text, (x + 25, y + 70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

keys =[["Q","W","E","R","T","Y","U","I","O","P"],
       ["A","S","D","F","G","H","J","K","L",";"],
       ["Z","X","C","V","B","N","M",",",".","BS"]]

button_width = 100
button_height = 100
x_gap = 15
y_gap = 20
x_start = 50
y_start = 50

buttonList = []
for row_index, row in enumerate(keys):
    y = y_start + row_index * 120
    for col_index, key in enumerate(row):
        x = x_start + col_index * (button_width + x_gap)
        buttonList.append(Button([x, y], key, size=[button_width, button_height]))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1288)
cap.set(4, 728)

finaltext = ""
lastPress = 0
cooldown = 0.4

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        h, w, c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    lmList.append([int(lm.x * w), int(lm.y * h)])

        distance = None
        if lmList:
            x1, y1 = lmList[8]
            x2, y2 = lmList[12]
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (46,46,46),3)
            cv2.circle(frame_bgr, (x1, y1), 8, (0,0,255), cv2.FILLED)
            cv2.circle(frame_bgr, (x2, y2), 8, (0,0,255), cv2.FILLED)

        for button in buttonList:
            hover = False
            if lmList and distance is not None and distance < 30:
                bx, by = button.pos
                bw, bh = button.size
                if bx < lmList[8][0] < bx + bw and by < lmList[8][1] < by + bh:
                    hover = True
                    if time.time() - lastPress > cooldown:
                        if button.text == "BS":
                            finaltext = finaltext[:-1]
                        else:
                            finaltext += button.text
                        lastPress = time.time()
            button.draw(frame_bgr, hover)

        cv2.rectangle(frame_bgr, (5, 550), (700, 450), (53, 53, 56), cv2.FILLED)
        cv2.putText(frame_bgr, finaltext, (60, 540), cv2.FONT_HERSHEY_PLAIN, 6, (255,255,255),5)

        cv2.imshow("Hand Tracking", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
