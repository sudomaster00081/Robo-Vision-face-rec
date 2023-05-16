import cv2

image_Path = 'Obama.jpg'
image = cv2.imread(image_Path)
# cv2.imshow('Output', image)
# cv2.imwrite('result.png', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# directory = r'C:\Users\APPK\Documents\COLLEGE\EXTRA\VINOD\Robo-Vision'

# print(image.shape)
image = cv2.resize(image, (800,800))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = "HELLO WORLD"
coordinates = (180,180)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 8, 8)
thickness = 2

image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness)

cv2.imshow('Output', image)

#cv2.imshow("Grayed", gray)
cv2.waitKey(0)