from imageai.Classification import ImageClassification
import os

for image_file in ["giraffe.jpg"]:  
    execution_path = os.getcwd()

    prediction = ImageClassification()
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
    prediction.loadModel()

    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, image_file), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)