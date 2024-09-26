## Make predictions and evaluate

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("\n Evaluation with Test Set\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Classification report
    print(classification_report(y_test, y_pred.round()))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred.round())
    print(cm)

    # Display Confusion Matrix
    class_arr = ['Legitimate', 'Fraud']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_arr)
    plt.figure(figsize=(12, 8))
    disp.plot(cmap='Reds', xticks_rotation='vertical', colorbar=False)
    plt.show()
