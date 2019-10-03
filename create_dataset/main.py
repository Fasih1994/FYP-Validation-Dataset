import cv2
import os

size = 260
height,width = 650,480

example_img_path = 'example.png'
example_img = cv2.imread(example_img_path)
example_img = cv2.resize(example_img,(height,width),interpolation=cv2.INTER_NEAREST)

### ----------------------- type the letter for which you are making the gesture-------------------//////////////
base_path = 'data'
name= 'Ahmed'
letter = 'y'
img_path  = os.path.join(base_path,name,letter)

start_point = ((height//2) - (size//2),(width//2) - (size//2))
end_point = (start_point[0]+size,start_point[1]+size)
print(start_point,end_point)
capture = cv2.VideoCapture(0)
while capture.isOpened():

    ret,frame = capture.read()
    if ret is True:
        cv2.imshow('Reference Image',example_img)
        cv2.rectangle(frame,start_point,end_point,(23,32,64),thickness=1)
        cv2.putText(frame,letter, (start_point[0],end_point[0]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255),3, lineType=cv2.LINE_AA)
        cv2.imshow('Video Frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('c'):
            if  not os.path.exists(img_path):
                os.makedirs(img_path)
            image = frame[start_point[1]+1:end_point[1],start_point[0]+1:end_point[0]]
            filename = '{}/{}_{}.png'.format(img_path,letter,len(os.listdir(img_path)))
            cv2.imwrite(filename=filename,img=image)
            print('image saved at ',filename)

        if cv2.waitKey(20) & 0xFF ==ord('q'):
            break
    else:
        break


capture.release()
cv2.destroyAllWindows()
