# HandwrittenDigitRecognition
HandWritten Digit Recognition using OpenCV and Neural_Networks
This program identifies handwritten digits using Neural networks. A classifier is trained with images in MNST database with 100
neurons in the hidden layer. The classifier is then saved in a separate file digits_cls.pkl to avoid training the classifier again and again as it is a time taking process.This process takes place in generateClassifier file.
Once the classifier is trained it can be used multiple times.
In performRecognition file the classifier is loaded first. Then the image to be tested is loaded. It is converted to grayscale and then binary. The image is the processed to find the contionous points. Recatangles are drawn around those points and then area inside each triangle is taken and the digit inside it is guessed.
