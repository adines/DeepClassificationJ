from ModelClassificationFactory import ModelClassificationFactory
import argparse

def predict(image, framework, model):
    modelo=ModelClassificationFactory.getClassificationModel(framework,model)
    return modelo.predict(image,model)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image to classify")
parser.add_argument("-f", "--framework", required=True, help="Framework used to classify the image")
parser.add_argument("-m", "--model", required=True, help="Model used to classify the image")

args=vars(parser.parse_args())
preds=predict(args["image"],args["framework"],args["model"])
print(preds)
