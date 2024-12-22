import csv
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Use a 70/30 split for training and testing
TEST_SIZE = 0.3

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    try:
        # Load the data from the provided CSV file
        evidence, labels = load_data(sys.argv[1])
    except FileNotFoundError:
        sys.exit("Error: File not found.")
    except Exception as e:
        sys.exit(f"Error loading data: {e}")

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42  # Fixed seed for reproducibility
    )

    # Initialize variables to track the best model
    best_k, best_threshold = 0, 0
    best_sensitivity, best_specificity = 0, 0

    print("\nSearching for the best k and threshold...")
    for k in range(1, 11):  # Testing k values from 1 to 10
        # Train the model with the current k
        model = train_model(X_train, y_train, k)

        # Perform cross-validation to evaluate the model's general performance
        scores = cross_val_score(model, evidence, labels, cv=5, scoring='accuracy')
        print(f"k = {k}, Cross-validation accuracy: {scores.mean():.2f}")

        # Get probabilities for test data
        probabilities = model.predict_proba(X_test)

        # Test various thresholds to find the best sensitivity and specificity balance
        for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
            # Adjust predictions based on the threshold
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            sensitivity, specificity = evaluate(y_test, predictions)

            print(f"k = {k}, Threshold = {threshold:.2f} -> Sensitivity: {100 * sensitivity:.2f}% | Specificity: {100 * specificity:.2f}%")

            # Update the best model if this configuration performs better
            if sensitivity + specificity > best_sensitivity + best_specificity:
                best_k = k
                best_threshold = threshold
                best_sensitivity = sensitivity
                best_specificity = specificity

    print("\nBest model found:")
    print(f"k = {best_k}, Threshold = {best_threshold:.2f}")
    print(f"Sensitivity: {100 * best_sensitivity:.2f}%")
    print(f"Specificity: {100 * best_specificity:.2f}%")

def load_data(filename):
    """
    Load data from a CSV file and return evidence and labels.
    Each row of evidence is a list of features, and labels are binary (1 if revenue, 0 otherwise).
    """
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    evidence = []
    labels = []

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract features and convert them to the appropriate types
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                months[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)

def train_model(evidence, labels, k=1):
    """
    Train a K-Nearest Neighbors model using the specified k.
    Uses distance-based weighting to improve performance.
    """
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Evaluate the model's performance by calculating sensitivity and specificity.
    Sensitivity: True Positive Rate
    Specificity: True Negative Rate
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
