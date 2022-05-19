from picamera.array import PiRGBArray
from picamera import PiCamera 
import cv2
from cv2 import moments
import time
import numpy as np
import RPi.GPIO as GPIO
#import board
#import adafruit_lis331

if __name__ == '__main__':
    def nothing(*arg):
        pass

#-----Borders------
left_bord = [0, 0]
right_bord = [0, 0]
up_bord = 0
R_find = False
L_find = False
y = 0
x_cam = 80
y_cam = 64
#-------------------


#i2c = board.I2C() 
#lis = adafruit_lis331.LIS331HH(i2c)
stop_time = 0 #для алгоритма поиска линии

#-----PD regulator-----
kp = 3.5
kd = 0.15
dt = 1
previous_time = 0
previous_fault = 0
#------------------

#----HSV start values----
h1 = 0
s1 = 0
v1 = 0
h2 = 255
s2 = 255
v2 = 255
#----------------------

#------Camera settings-------
lengh = 160
weidh = 128 
camera = PiCamera() #Инициализация камеры
camera.resolution = (lengh, weidh) #размер окна
#camera.rotation = 180 #Поворот камеры на 180 градусов
camera.framerate = 32 #Выбор частоты кадров
raw_capture = PiRGBArray(camera, size=(lengh, weidh)) #Зават изобраения с камеры
time.sleep(0.01) #Минималная пауза
#-----------------------------

#------GPIO settings---------
GPIO.setmode(GPIO.BOARD)
pins = [7, 11, 12, 13, 15, 16]
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
pwm_1 = GPIO.PWM(12, 1000)
pwm_2 = GPIO.PWM(16, 1000)
dutyCycle = 22
duty_plus = dutyCycle
pwm_1.start(dutyCycle)
pwm_2.start(dutyCycle)
#------------------------------

#------------------------OpenCV settings-----------------
cv2.namedWindow( "result" ) # создаем главное окно
cv2.namedWindow( "settings" ) # создаем окно настроек
cap = cv2.VideoCapture(0)
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
#-----------------------------------------------------------

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True): 
    #подбор h2
    img = raw_capture.array
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv2.inRange(hsv, h_min, h_max)
    while thresh[64,80] != 0:
        h2 -= 1
        img = raw_capture.array
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)
        thresh = cv2.inRange(hsv, h_min, h_max)
        print(h2)
    if h2 < 247:
        h2 += 12
    else:
        h2 = 255
    img = raw_capture.array
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv2.inRange(hsv, h_min, h_max)
    #подбор h1
    while thresh[64,80] != 0:
        h1 += 1
        img = raw_capture.array
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)
        thresh = cv2.inRange(hsv, h_min, h_max)
        print(h1)
    if h1 > 8:    
        h1 -= 12
    else:
        h1 = 0
    img = raw_capture.array
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv2.inRange(hsv, h_min, h_max)
    #подбор s1
    while thresh[64,80] != 0:
        s1 += 1
        img = raw_capture.array
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)
        thresh = cv2.inRange(hsv, h_min, h_max)
        print(s1)
    if s1 > 20:
        s1 -= 75
    else:
        s1 = 0
    img = raw_capture.array
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv2.inRange(hsv, h_min, h_max)
    #подбор v1
    while thresh[64,80] != 0:
        v1 += 1
        img = raw_capture.array
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)
        thresh = cv2.inRange(hsv, h_min, h_max)
        print(v1)
    if v1 > 90:
        v1 -= 90
    else:
        v1 = 0
    print('done!')
    raw_capture.truncate(0)
    break

cv2.setTrackbarPos('h1', 'settings',h1)
cv2.setTrackbarPos('s1', 'settings', s1)
cv2.setTrackbarPos('v1', 'settings', v1)
cv2.setTrackbarPos('h2', 'settings', h2)

start_time = time.time()

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    img = raw_capture.array
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsv, h_min, h_max)
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']

    if dArea > 100:
        x_circ = int(dM10 / dArea)
        y_circ = int(dM01 / dArea)
        #print(f"x: {x_circ}, y: {y_circ}")
        cv2.circle(img, (x_circ, y_circ), 10, (0,0,255), -1)
        y_L = 0
        x_L = 0
        if thresh[y_L, x_L] == 0:
            L_find = False
            while x_L != 157 and L_find == False:
                while y_L != 129:
                    if y_L != 128 and thresh[y_L, x_L] != 0:
                        left_bord = [y_L, x_L]
                        L_find = True
                        break
                    elif y_L == 128:
                        y_L = 0
                        break
                    else:
                        y_L += 1
                x_L += 1
        y_R = 0
        x_R = 159
        if thresh[y_R, x_R] == 0:
            R_find = False
            while x_R != 0 and R_find == False:
                while y_R != 129:
                    if y_R != 128 and thresh[y_R, x_R] != 0:
                        right_bord = [y_R, x_R]
                        R_find = True
                        break
                    elif y_R == 128:
                        y_R = 0
                        break
                    else:
                        y_R += 1
                x_R -= 1
            up_bord = 0
            if  thresh[up_bord, x_circ] == 0:
                while thresh[up_bord, x_circ] == 0 and up_bord != 127:
                    up_bord+=1

    
        fault = abs(x_cam - x_circ)
        duty_aman = (kp*fault+kd*(fault-previous_fault)/dt)/10
        duty_plus = (dutyCycle + duty_aman)
        if duty_plus > 50:
            duty_plus = 50
        elif duty_plus < 38:
            duty_plus = 38

        previous_fault = fault

        if (x_circ - x_cam > 42):
            stop_time = 0
            pwm_1.ChangeDutyCycle(duty_plus)
            pwm_2.ChangeDutyCycle(duty_plus)
            GPIO.output(11, GPIO.HIGH)
            GPIO.output(7, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(13, GPIO.HIGH)
            print("moving right    / ", duty_plus)

        elif (x_cam - x_circ > 42):
            stop_time = 0 
            pwm_1.ChangeDutyCycle(duty_plus)
            pwm_2.ChangeDutyCycle(duty_plus)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(7, GPIO.HIGH)
            GPIO.output(15, GPIO.HIGH)
            GPIO.output(11, GPIO.LOW)
            print("moving LEFT    / ", duty_plus)

        elif ((x_circ - x_cam < 42) and (x_cam - x_circ < 42)) and (right_bord[0] > 64 and right_bord[1] > 140):
            stop_time = 0
            pwm_1.ChangeDutyCycle(50)
            pwm_2.ChangeDutyCycle(50)
            GPIO.output(11, GPIO.HIGH)
            GPIO.output(7, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(13, GPIO.HIGH)
            print("moving ultra right    / ", 42)

        elif (x_circ - x_cam < 42) and (x_cam - x_circ < 42):
            stop_time = 0
            pwm_1.ChangeDutyCycle(dutyCycle)
            pwm_2.ChangeDutyCycle(dutyCycle)
            GPIO.output(11, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(7, GPIO.HIGH)
            GPIO.output(13, GPIO.HIGH)
            print("moving STRAIGHT    / ", dutyCycle)
    
        
        real_time = round((time.time() - start_time), 2)
        previous_time = real_time - previous_time
        dt = previous_time
    elif dArea <= 100 and (stop_time == 0):
        pwm_1.ChangeDutyCycle(dutyCycle)
        pwm_2.ChangeDutyCycle(dutyCycle)
        GPIO.output(7, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(13, GPIO.LOW)
        print("moving backward    / ", dutyCycle)
        stop_time = time.perf_counter()
        

    elif dArea <= 100 and time.perf_counter() - stop_time > 1 and time.perf_counter() - stop_time < 2:
        pwm_1.ChangeDutyCycle(38)
        pwm_2.ChangeDutyCycle(38)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(7, GPIO.HIGH)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(11, GPIO.LOW)
        print("moving LEFT    / ", dutyCycle)
        

    elif dArea <= 100 and time.perf_counter() - stop_time > 2 and time.perf_counter() - stop_time < 4:
        pwm_1.ChangeDutyCycle(42)
        pwm_2.ChangeDutyCycle(42)
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(7, GPIO.LOW)
        GPIO.output(15, GPIO.LOW)
        GPIO.output(13, GPIO.HIGH)
        print("moving RIGHT    / ", 42)
    
    elif dArea <= 100 and time.perf_counter() - stop_time > 4 and time.perf_counter() - stop_time < 5:
        pwm_1.ChangeDutyCycle(38)
        pwm_2.ChangeDutyCycle(38)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(7, GPIO.HIGH)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(11, GPIO.LOW)
        print("moving LEFT    / ", 38)
        
        

    elif dArea <= 100 and time.perf_counter() - stop_time > 5:
        stop_time = 0

    print(stop_time)
    #print('right_y: ',right_bord[0],'  /     right_x: ', right_bord[1])
    #print("Acceleration : X: %.i, Y:%.i, Z:%.i ms^2" % lis.acceleration)

    cv2.imshow('result', img) 
    cv2.imshow('thresh', thresh)
    ch = cv2.waitKey(5)
    raw_capture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
pwm_1.stop()
pwm_2.stop()
GPIO.cleanup()
print("\nstop")
cap.release()
cv2.destroyAllWindows()

