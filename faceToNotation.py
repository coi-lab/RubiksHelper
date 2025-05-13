import cv2


def draw_and_sample_grid(frame):
    h, w = frame.shape[:2]
    box_size = min(h, w) // 2
    start_x = (w - box_size) // 2
    start_y = (h - box_size) // 2
    cell_size = box_size // 3

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_colors = []

    for row in range(3):
        row_colors = []
        for col in range(3):
            cx = start_x + col * cell_size + cell_size // 2
            cy = start_y + row * cell_size + cell_size // 2

            # Extract a 10x10 patch around the center
            patch = hsv_frame[cy-5:cy+5, cx-5:cx+5]
            avg_hsv = cv2.mean(patch)[:3]  # (H, S, V)

            color_label = classify_color(avg_hsv)
            row_colors.append(color_label)

            # Draw the sampling circle and label
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(frame, color_label[0].upper(), (cx - 5, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw grid lines
        face_colors.append(row_colors)

    return frame, face_colors


def classify_color(hsv):
    h, s, v = hsv
    if v < 50:
        return 'black'
    elif s < 50 and v > 200:
        return 'white'
    elif 0 <= h < 10 or h > 160:
        return 'red'
    elif 10 <= h < 25:
        return 'orange'
    elif 25 <= h < 35:
        return 'yellow'
    elif 35 <= h < 85:
        return 'green'
    elif 85 <= h < 130:
        return 'blue'
    else:
        return 'unknown'


# video capture with a grid
def main():
    # video capture opens a video stream, 0 refers to the default webcam
    cap = cv2.VideoCapture(0)
    # infinite loopto keep readin and showing frames
    while True:
        # frame is the actual image captured from the camera
        # captures one frame from the camera
        ret, frame = cap.read()
        frame, sampled_colors = draw_and_sample_grid(frame)
        # if no frame is captured, break the loop
        if not ret:
            break
            print("Failed to capture image")
        # displays the captured frame in a window
        cv2.imshow("Cube Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Captured colors:", sampled_colors)  # per face
        # loops breaks once q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # closes the capturing device
    cap.release()
    cv2.destroyAllWindows()


main()
