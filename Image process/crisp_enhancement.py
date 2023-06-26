import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:\\desk\\Medical image\\Brain Tumor MRI Dataset\\Training\\glioma\\origin\\Tr-gl_0010.jpg', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobelCom = cv2.bitwise_or(sobelx, sobely)

C = img+laplacian
E = cv2.blur(sobelCom, (5, 5))
F = C*E

mina = np.min(F)
maxa = np.max(F)
#print(mina,maxa)
F = np.uint8(255*(F-mina)/(maxa-mina))
G = img+F
H = cv2.pow(G/255.0, 0.5)

plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray')
plt.title('A = original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 2), plt.imshow(laplacian, cmap='gray')
plt.title('B = laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 3), plt.imshow(C, cmap='gray')
plt.title('C = add a and b'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 4), plt.imshow(sobelCom, cmap='gray')
plt.title('D = sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 5), plt.imshow(E, cmap='gray')
plt.title('E = blur sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 6), plt.imshow(F, cmap='gray')
plt.title('F = C*E'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 7), plt.imshow(G, cmap='gray')
plt.title('G = add a and f'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 8), plt.imshow(H, cmap='gray')
plt.title('H = mi lv'), plt.xticks([]), plt.yticks([])

plt.show()
