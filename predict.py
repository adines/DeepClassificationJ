from ModelClassificationFactory import ModelClassificationFactory
import argparse

# Metodo de la API para predecir la clase a la que pertenece una imagen
# image: Imagen que se desea clasificar
# framework: framework que se desea utilizar para clasificar la imagen
# model: Modelo que se desea utilizar para clasificar la imagen
# Salida: Tupla clase, probabilidad
def predict(image, framework, model):
    modelo=ModelClassificationFactory.getClassificationModel(framework,model)
    return modelo.predict(image,model)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image to classify")
parser.add_argument("-f", "--framework", required=True, help="Framework used to classify the image")
parser.add_argument("-m", "--model", required=True, help="Model used to classify the image")

args=vars(parser.parse_args())
# print(predict(args["image"],args["framework"],args["model"]))
preds=predict(args["image"],args["framework"],args["model"])
print(preds)
