
import os
import imutils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import random
from imutils import contours
import matplotlib
matplotlib.use('tkagg')


# ========================================================================
# Este codigo resume la parte de la inferencia(usando modelos) como punto final, es decir
# solamante cargamos y ejecutamos los modelos previamente implementados y
# entrenados en google colab pro. esto es solo para propositos de demostra-
# cion.
# ========================================================================

# Cargar el modelo pre-entranado YOLO, con el peso y la configuración adecuada
net = cv2.dnn.readNet("models/trained_display/yolov3_training_last.weights",
                      "models/trained_display/yolov3_testing.cfg")

# Nombre de la classe
classes = ["meter"]


# Resize images
width = 500
height = 650
dim = (width, height)

# images_path = glob.glob(r"display_input/03.jpg")
# Lectura de todos los archivos .jpg(todas la images en el folder)
list_images_path = glob.glob(r"display_input/*.jpg")
print(" la lista de images es:")
print(list_images_path)

# Seleccionamos aleatoriamente una imagen
list_images_path = [random.choice(list_images_path)]
# Mostrar imagen elegida
print("image choice :", list_images_path[0])
# Mostrar la imagen 01
img_test = cv2.imread(list_images_path[0])
# Convertir imagen de  BGR a RGB (formato)
img_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
img_test = plt.imshow(img_rgb)
plt.show()

# --------------------------------------------------------------------------------

# instanciamos a las capas para obtener la ultima de YOLO
layer_names = net.getLayerNames()

# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#  elegimos la capa de salida para poder predecir
# the net returns integers,no iterable use this instead.
display_output_layers = [layer_names[i-1]
                         for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

random.shuffle(list_images_path)

CROPPED_IMAGE = ""
for img_path in list_images_path:
    # Loading image
    img = cv2.imread(img_path)  # leer imagen
    img = cv2.resize(img, dim)  # escalar imagen 500, 650
    height, width, channels = img.shape
    # print(height, width,channels)
    # Adapatando input para yolo trained
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)  # ingregar imaen
    outs = net.forward(display_output_layers)  # salida de red

    # mostramos el output
    # print("outss:",outs)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # elegir solo los maximos top
            confidence = scores[class_id]
            # print("confidence: ",confidence)
            if confidence > 0.3:
                # objectos detectados
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # coordenadas de rectangulos
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # print(x, y, w, h)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(confidences[0])
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[class_ids[i]]
            # color verde
            color_verde = (123, 236, 73)  # (R,G,B)
            cv2.rectangle(img, (x, y), (x + w, y + h), color_verde, 2)
    # recortar la imagen
    # crop_img = img[y:y+h, x:x+w]
    # print(crop_img)
    img2 = plt.imshow(img)
    plt.show()
    print(y, h, x, w)
    crop_img = img[y-20:y+h+20, x:x+w]
    print("PLOT IMAGE FROM CROP")
    crop_img = plt.imshow(crop_img)
    plt.show()
    # cv2_imshow(img)
    # guardamos la imagen recortada dentro del drive
    crop_img_name = os.path.basename(img_path)
    CROPPED_IMAGE = crop_img_name
    cv2.imwrite("display_output/display_out_" + crop_img_name,
                img[y-20:y+h+20, x:x+w])

# ---------------------------------------- DIGIT RECOGNITION --------------------


# Cargar el modelo entrenado , ocupa aprox 4 gb
digit_net = cv2.dnn.readNet("models/trained_digit/digit_recognition.weights",
                            "models/trained_digit/digit_recognition.cfg")

# instanciamos a las capas para obtener la ultima de YOLO
layer_names = digit_net.getLayerNames()

#  elegimos la capa de salida para poder predecir
# the net returns integers,no iterable use this inste
digit_output_layers = [layer_names[i-1]
                       for i in digit_net.getUnconnectedOutLayers()]

# -----------------------------------------------------------
# SHOW THE IMAGE (DISPLAY CROPPED)
# -----------------------------------------------------------
# Resize images de acuerdo a lo que sale
width_digit = 200
height_digit = 80
classes_digit = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
dim2 = (width_digit, height_digit)

# cargamos las imagenes recortadas
# images_path_digit = glob.glob(r"display_output/crop_01.jpg")
images_path_digit = glob.glob(r"display_output/display_out_"+CROPPED_IMAGE)
print(images_path_digit)

img_test = cv2.imread(images_path_digit[0])
# Convert the image from BGR to RGB format
img_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

img_test = plt.imshow(img_rgb)
plt.show()

# Se definen los colores a usar ramdon para cada digito

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.figure(figsize=(10, 12))
current_axis = plt.gca()

for img_path in images_path_digit:

    # Loading image
    img_cropped = cv2.imread(img_path)
    img_1 = cv2.resize(img_cropped, dim2)
    # img = orig_images[0]
    height, width, channels = img_1.shape
    # print(height, width,channels)
    # detectando objeto
    blob = cv2.dnn.blobFromImage(
        img_1, 0.00392, (300, 300), (0, 0, 0), True, crop=False)

    digit_net.setInput(blob)
    outs = digit_net.forward(digit_output_layers)

    # mostramos el output
    # print("outss:",outs)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # escojer solo los maximos top
            confidence = scores[class_id]
            # print("confidence: ",confidence)
            if confidence > 0.3:
                # objectos detectados
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # coordenadas de rectangulos
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # print(x, y, w, h)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(confidences[0])
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 0.9
    thickness = 1

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes_digit[class_ids[i]])
            # print("digito: {}".format(label))
            # color = colors[class_ids[i]]
            color = colors[class_ids[i]]
            label = 'díg {}: {:.2f} %'.format(label, confidences[i] * 100)
            print(label)
            print()

            current_axis.add_patch(plt.Rectangle(
                (x, y), w, h, color=color, fill=False, linewidth=2))
            current_axis.text(x, y, label, size='x-large',
                              color='white', bbox={'facecolor': color, 'alpha': 0.9})

    # recortar la imagen & guardar en test_amr
    # plt.figure(figsize=(10,10))
    plt.imshow(img_1)
    # plt.show()

    # Guardar la Imagen
    # lugar donde de guarda : path_location
    save_path = 'digit_output/digit_recognition_01.jpg'
    plt.savefig(save_path)  # , bbox_inches='tight', pad_inches=0)
    print(f"Image saved to {save_path}")

    # Mostrar plotteo
    plt.show()
