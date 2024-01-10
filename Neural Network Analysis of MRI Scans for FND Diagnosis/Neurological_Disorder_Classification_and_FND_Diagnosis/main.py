import os
import pandas as pds
import numpy as npy
import matplotlib.pyplot as mpl
import seaborn as sbn
from scipy import signal
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
def read_data(file):
    data = []
    df = pds.read_csv(file)
    data.append(df)
    return pds.concat(data)



def filter_data(data, high_pass_cutoff=0.5, low_pass_cutoff=35):
    # Apply high-pass filter if the data length exceeds the threshold.
    if len(data) > 8001:
        # Design a high-pass filter
        high_pass_coeffs = signal.butter(N=4, Wn=high_pass_cutoff, btype='high', analog=False)
        # Pad the signal to compensate for the filter delay
        padded_signal = npy.pad(data[:, 1], (0, sum(len(coef) for coef in high_pass_coeffs) - 2), mode='constant')
        # Apply the high-pass filter
        data[:, 1] = signal.filtfilt(*high_pass_coeffs, padded_signal)[len(high_pass_coeffs[0]) - 1 : -len(high_pass_coeffs[1]) + 1]

    # Apply low-pass filter to the (now high-pass-filtered) signal
    low_pass_coeffs = signal.butter(N=4, Wn=low_pass_cutoff, btype='low', analog=False)
    data[:, 1] = signal.filtfilt(*low_pass_coeffs, data[:, 1])
    return data



def partition_dataset(features, target, test_ratio=0.2, validation_ratio=0.25):
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=test_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
def display_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    mpl.figure(figsize=(8, 6))
    sbn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    mpl.title(f"Confusion Matrix - {model_name}")
    mpl.ylabel('Actual')
    mpl.xlabel('Predicted')
    mpl.show()





def display_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pds.DataFrame(report).round(2)
    print(f"Classification Report - {model_name}:\n", df_report)
def display_heatmaps(report, model_name):
    precision_recall = npy.array([report['0']['precision'], report['1']['precision'], report['0']['recall'], report['1']['recall']])
    accuracy_macro_avg_weighted_avg = npy.array([report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], 
                                                 report['macro avg']['f1-score'], report['weighted avg']['precision'], 
                                                 report['weighted avg']['recall'], report['weighted avg']['f1-score']])
    fig, axes = mpl.subplots(1, 2, figsize=(15, 6))
    sbn.heatmap(precision_recall.reshape(2, 2), annot=True, cmap="Blues", xticklabels=["Precision 0", "Precision 1", "Recall 0", "Recall 1"], 
                yticklabels=["Actual 0", "Actual 1"], fmt=".2f", ax=axes[0])
    axes[0].set_title(f"Heatmap for Precision and Recall - {model_name}")
    xticklabels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Weighted Precision", "Weighted Recall", "Weighted F1-Score"]
    sbn.heatmap(accuracy_macro_avg_weighted_avg.reshape(1, -1), annot=True, cmap="Blues", xticklabels=xticklabels, fmt=".2f", ax=axes[1])
    axes[1].set_title(f"Heatmap for Accuracy, Macro Avg, Weighted Avg - {model_name}")
    mpl.show()
def visualize_activation_patterns(model, layer_name, input_data):
    # Get the intermediate layer output
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = intermediate_layer_model.predict(input_data)
    # Ensure activations are 4D
    if activations.ndim == 2:
        # Add dimensions to make it 4D for individual activations
        activations = activations.reshape(activations.shape[0], activations.shape[1], 1, 1)
    # Calculate the number of grid rows and columns
    num_activations = activations.shape[-1]
    num_cols = int(npy.ceil(npy.sqrt(num_activations)))
    num_rows = int(npy.ceil(num_activations / num_cols))
    # Set up the figure
    fig, axes = mpl.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # Ensure axes is an array even if there is only one subplot
    if num_activations == 1:
        axes = npy.array([axes])
    # Flatten axes array
    axes_flat = axes.flatten()
    # Plot each activation
    for i in range(num_activations):
        ax = axes_flat[i]
        activation = activations[0, :, :, i].squeeze()
        if activation.ndim == 1:
            activation = activation.reshape(int(npy.sqrt(activation.shape[0])), -1)
        im = ax.imshow(activation, cmap='viridis', interpolation='nearest', aspect='auto')
        ax.set_title(f'Activation {i + 1}')
        ax.axis('off')
    # Hide any unused axes
    for i in range(num_activations, len(axes_flat)):
        axes_flat[i].axis('off')
    # Add color bar
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    mpl.show()
def classify_neurological_disorder(training_features, training_labels, testing_features, testing_labels):
   # Normalize the feature space
    feature_scaler = StandardScaler()
    normalized_training_features = feature_scaler.fit_transform(training_features)
    normalized_testing_features = feature_scaler.transform(testing_features)
    # Initialize and train the logistic regression model
    disorder_predictor = LogisticRegression()
    disorder_predictor.fit(normalized_training_features, training_labels)
    # Predict the disorders using the testing set
    disorder_predictions = disorder_predictor.predict(normalized_testing_features)
    # Compile a report on the model's performance
    performance_report = classification_report(testing_labels, disorder_predictions, output_dict=True)
    # Visualizations
    model_title = "Logistic Regression - Neurological Disorder Prediction"
    display_confusion_matrix(testing_labels, disorder_predictions, model_title)
    display_classification_report(testing_labels, disorder_predictions, model_title)
    display_heatmaps(performance_report, model_title)
def perform_t_test(y_true, y_pred_probs):
    # Calculate the t-statistic and the p-value
    t_stat, p_val = ttest_ind(y_pred_probs, y_true)
    print(f"T-test result: t-statistic={t_stat:.2f}, p-value={p_val:.3f}")
    # Plot the histogram of predicted probabilities and actual labels
    mpl.figure(figsize=(10, 6))
    sbn.histplot(y_pred_probs, color="blue", label="Predicted Probabilities", kde=False)
    sbn.histplot(y_true, color="red", label="Actual Labels", kde=False, bins=[-0.5, 0.5, 1.5])
    # Plot the means of both distributions
    mpl.axvline(npy.mean(y_pred_probs), color="blue", linestyle='--')
    mpl.axvline(npy.mean(y_true), color="red", linestyle='--')
    # Annotate the t-statistic and p-value in the plot
    mpl.text(0.5, max(mpl.ylim()), f't-statistic={t_stat:.2f}', ha='center', va='bottom', color="blue")
    mpl.text(0.5, max(mpl.ylim()), f'p-value={p_val:.3f}', ha='center', va='top', color="red")
    # Add the legend, title, and show the plot
    mpl.legend()
    mpl.title('T-test Comparison')
    mpl.xlabel('Value')
    mpl.ylabel('Frequency')
    mpl.show()



def display_roc_curve(y_true, y_pred_probs, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    mpl.figure(figsize=(8, 6))
    mpl.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    mpl.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    mpl.xlim([0.0, 1.0])
    mpl.ylim([0.0, 1.05])
    mpl.xlabel('False Positive Rate')
    mpl.ylabel('True Positive Rate')
    mpl.title(f'Receiver Operating Characteristic - {model_name}')
    mpl.legend(loc="lower right")
    mpl.show()


def classify_FND(train_data, train_labels, test_data, test_labels, validation_data, validation_labels):
    neural_network = Sequential([
        Dense(units=64, activation='relu', input_dim=train_data.shape[1]),
        Dense(units=64, activation='relu'),
        Flatten(),
        Dense(units=1, activation='sigmoid')
    ])
    # Compile the model with loss and optimizer
    neural_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model with the training data and validation data
    neural_network.fit(x=train_data, y=train_labels, validation_data=(validation_data, validation_labels), epochs=10, verbose=0)
    # Predict probabilities on the test set
    predictions_prob = neural_network.predict(test_data).flatten()
    # Convert probabilities to binary predictions
    predictions_binary = (predictions_prob > 0.5).astype(int)
    # Perform and display the t-test
    print("Performing t-test between actual labels and predicted probabilities...")
    perform_t_test(test_labels, predictions_prob)
    # Perform and display the ROC curve analysis
    print("Displaying ROC curve...")
    display_roc_curve(test_labels, predictions_prob, "FND Neural Network")
    report = classification_report(test_labels, predictions_prob, output_dict=True)
    display_confusion_matrix(test_labels, predictions_prob, "FND Neural Network")
    display_classification_report(test_labels, predictions_prob, "FND Neural Network")
    display_heatmaps(report, "FND Neural Network")
    # Visualize activation patterns for the first hidden layer
    visualize_activation_patterns(neural_network, 'dense_30', test_data)
    return test_labels, predictions_prob



def FND_diagnosis_statement(y_test, y_pred_proba):
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    diagnosis = "The patient is diagnosed with FND." if roc_auc >= 0.7 else "The patient is not diagnosed with FND."
    return diagnosis, roc_auc
def main():
    file = "C:\\Users\\unite\\OneDrive\\Desktop\\research\\Neurological_Disorder_Classification_and_FND_Diagnosis\\csv"
    data = read_data(file)
    filtered_data = filter_data(data)
    labels = npy.array([0 if i < len(filtered_data) / 2 else 1 for i in range(len(filtered_data))])
    data_npy = filtered_data.to_numpy()
    X_train, X_test, X_val, y_train, y_test, y_val = partition_dataset(data_npy, labels)
    print("Classifying neurological disorder:")
    classify_neurological_disorder(X_train, y_train, X_test, y_test)
    print("Classifying FND with Neural Network:")
    y_test_nn, y_pred_nn = classify_FND(X_train, y_train, X_test, y_test, X_val, y_val)
    diagnosis, roc_auc = FND_diagnosis_statement(y_test_nn, y_pred_nn)
    print("\nDiagnostic Statement:")
    print(diagnosis)
    print("ROC AUC: {:.2f}".format(roc_auc))
if __name__ == '__main__':
    main()



