import cv2
import os

size = 260
height,width = 650,480

example_img_path = 'example.png'
example_img = cv2.imread(example_img_path)
example_img = cv2.resize(example_img,(height,width),interpolation=cv2.INTER_NEAREST)

### ----------------------- type the letter for which you are making the gesture-------------------//////////////
base_path = 'data'
name= 'Shehryar'
base_path  = os.path.join(base_path,name)
if not os.path.exists(base_path):
    os.mkdir(base_path)
alphs = "a b c d e f g h i k l m n o p q r s t u v w x y".split()
start_point = ((height//2) - (size//2),(width//2) - (size//2))
end_point = (start_point[0]+size,start_point[1]+size)
print(start_point,end_point)
capture = cv2.VideoCapture(0)
while capture.isOpened():

    ret,frame = capture.read()
    if ret is True:
        cv2.imshow('Reference Image',example_img)
        cv2.rectangle(frame,start_point,end_point,(23,32,64),thickness=1)
        cv2.imshow('Video Frame',frame)
        key = cv2.waitKey(1)
        if  key != -1:
            key = chr(key)
            print(key)
            if key !='j':
                img_path = os.path.join(base_path,key)
                if  not os.path.exists(img_path):
                    os.makedirs(img_path)
                image = frame[start_point[1]+1:end_point[1],start_point[0]+1:end_point[0]]
                filename = '{}/{}_{}.png'.format(img_path,key,len(os.listdir(img_path)))
                cv2.imwrite(filename=filename,img=image)
                print('image saved at ',filename)
            else:
                break

    else:
        break


capture.release()
cv2.destroyAllWindows()
