import os
import cv2


def create_person_directory(person_name):
    directory = os.path.join("images", person_name.lower().replace(" ", "_"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def capture_images(directory):
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                cv2.imshow("Captured Image", face_img)

                filename = os.path.join(directory, f"image{count+1}.jpg")
                cv2.imwrite(filename, face_img)
                count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    person_name = input("Enter the person's name: ")
    directory = create_person_directory(person_name)
    capture_images(directory)


if __name__ == "__main__":
    main()
