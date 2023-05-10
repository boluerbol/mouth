import dlib
import cv2
import numpy as np
import mediapipe as mp
import time
import speech_recognition 
sr = speech_recognition.Recognizer()
sr.pause_threshold = 0.5

def create_task():
    ''' Create a todo task '''
    print("  Микрофон включен  ")
    with speech_recognition.Microphone() as mic:
        sr.adjust_for_ambient_noise(source=mic, duration=0.5)
        audio = sr.listen(source=mic)
        query = sr.recognize_google(audio_data=audio, language='ru-RU').lower()
    with open("erbol.txt","a", encoding="utf-8") as file:
        file.write(f'# {query}\n')
    print((query.upper()))
    return f'Задача {query} добавлена в todo-list! '
cap = cv2.VideoCapture(0)
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
# Создаем объекты для обнаружения лица и точек на лице с помощью библиотеки dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Основной цикл программы
while True:
    try:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
        # Читаем кадр из видеопотока
        ret, frame = cap.read()
        if not ret:
            break        
        # Обнаруживаем лица на кадре
        faces = detector(frame)       
        # Если лица обнаружены, то выполняем следующие действия
        if len(faces) > 0:
            # Выбираем первое обнаруженное лицо
            face = faces[0]
            
            # Определяем координаты губ на лице
            landmarks = predictor(frame, face)
            lip_top1 = (landmarks.part(62).x, landmarks.part(62).y)
            lip_bottom1 = (landmarks.part(66).x, landmarks.part(66).y)
            
            # Определяем расстояние между губами
            lip_distance1 = np.sqrt((lip_bottom1[0] + lip_bottom1[1])*2 - (lip_top1[0]+lip_top1[1])*2)
            
            if  lip_distance1 > 5 :
                print("yes")
                with speech_recognition.Microphone() as mic:
                    print(create_task())
            else:
                print("No")
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f'FPS: {int(fps)}' ,(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("IMage", img)
        cv2.waitKey(1)
    except speech_recognition.exceptions.UnknownValueError:
        print("Не понятная речь!!!")
        continue