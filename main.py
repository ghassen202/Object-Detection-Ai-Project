import cv2
import numpy as np

image_path= 'D:\AI Project\Models\gathering.jpg'
prototxt_path= 'D:\AI Project\Models\MobileNetSSD_deploy.prototxt.txt'
model_path= 'D:\AI Project\Models\MobileNetSSD_deploy.caffemodel'
min_confidence= 0.2

classes=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0,255,size=(len(classes), 3))
# Load the network
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# Load the input image
#image = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
while True:
############### Block instead of blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300),0.007,(300,300),130) ################
    _,image = cap.read()
    height, width =image.shape[0], image.shape[1]
    # Define the input shape of the network
    input_shape = (300, 300)
    # Resize the image to the input shape
    resized_image = cv2.resize(image, input_shape)
    # Subtract the mean values of the dataset from the image
    mean = (104, 117, 123) # Mean values for MobileNet SSD
    mean_subtracted_image = resized_image - mean
    # Scale the pixel values to the range [-1, 1]
    scaled_image = mean_subtracted_image / 127.5
    scaled_image -= 1.0
    # Transpose the dimensions of the image to match the network's input shape
    transposed_image = np.transpose(scaled_image, (2, 0, 1))
    # Reshape the image to a 4D blob
    blob = np.expand_dims(transposed_image, axis=0)
##################################################################################################################################

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network
    detected_objects = net.forward()
    print(detected_objects[0][0][1])

    for i in range(detected_objects.shape[2]):
        confidence=detected_objects[0][0][i][2]
        if confidence > min_confidence:

            class_index= int(detected_objects[0,0,i,1])
            upper_left_x = int(detected_objects[0,0,i,3] * width)
            upper_left_y = int(detected_objects[0,0,i,4] * height)
            lower_right_x = int(detected_objects[0,0,i,5] * width)
            lower_right_y = int(detected_objects[0,0,i,6] * height)

            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.rectangle(image,(upper_left_x,upper_left_y),(lower_right_x,lower_right_y),
                        colors[class_index], 1)
            cv2.putText(image,prediction_text,(upper_left_x,upper_left_y-15 if upper_left_y > 30 else upper_left_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)
            
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", cv2.resize(image, (800, 600)))
    cv2.waitKey(5)
cv2.destroyAllWindows()
cap.release()
