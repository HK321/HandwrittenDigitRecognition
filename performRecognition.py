# Import the modules-r4
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import cv2
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
from tkinter.filedialog import askopenfile
# Load the classifier
clf = joblib.load("digits_cls.pkl")
global filename
global num
# Read the input image
def load_file():
    global filename

    text_file= tkinter.filedialog.askopenfilename(filetypes=[('All Files' , '')])
    filename = text_file.split("/")
    filename=filename[len(filename)-1]

#produce output
def produce_result():
    global filename
    global num
    num=''
    number=[]
    if filename==None:
        tkinter.messagebox.showerror(title="Error", message="Please select a F-ile Before Recognition" )

    im = cv2.imread(filename,1)
    # Convert to grayscale and apply Gaussian filtering
    # Convert image from one color space to another
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    # Threshold to binary the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    # Find contours in the image
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        if(pt1<0):
            pt1=0
        if(pt2<0):
            pt2=0
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        number.append(nbr[0])
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        print (nbr[0])
    num = ''.join(str(e) for e in number)
    cv2.imshow("Resulting Image with Rectangular ROIs :"+num, im)
    cv2.waitKey()
#create menu to take image input
root = tkinter.Tk()
root.title("DigitalRecognition")
root.minsize(width=200,height=200)
root.maxsize(width=500,height=500)
loadLabel = tkinter.Label(root, text="Digital Recognition System")
footerLabel= tkinter.Label(root, text="Powered By :")
loadImage= tkinter.Button(root, text="Choose Image", command=load_file)
loadRecognition = tkinter.Button(root, text="Recognize" ,command=produce_result)
loadLabel.pack()
loadImage.pack()
print("\n\n\n")
loadRecognition.pack()
footerLabel.pack(side=BOTTOM)
root.mainloop()