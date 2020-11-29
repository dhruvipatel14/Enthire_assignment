from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
import pickle

class MakePrediction:
    def __init__(self,x_test,y_test):
        self.x_test = x_test
        self.y_test = y_test

    def predications(self,model_path):
        model = pickle.load(open(model_path, 'rb'))
        predictions = model.predict(self.x_test)

        print("Accuracy :", metrics.accuracy_score(self.y_test, predictions))

        precision = precision_score(self.y_test, predictions, average='micro')
        recall = recall_score(self.y_test, predictions, average='micro')
        f1 = f1_score(self.y_test, predictions, average='micro')

        print("Micro-averasge quality numbers")
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(
            precision, recall, f1))

        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        print("Macro-average quality numbers")
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(
            precision, recall, f1))

        print(metrics.classification_report(self.y_test, predictions))
