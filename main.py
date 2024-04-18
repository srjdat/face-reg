import cv2
from simple_facerec import SimpleFacerec


sfr = SimpleFacerec()
sfr.load_encoding_images('faces/')

process_this_frame = True

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 255), -1)
        cv2.putText(frame, name, (right - 4, bottom -6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()