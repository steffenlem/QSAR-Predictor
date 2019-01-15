# import all required libraries here
import sys
import getopt
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AllChem import GetHashedAtomPairFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
from xgboost import XGBClassifier

from sklearn.metrics import *

__author__ = 'Steffen Lemke & Thomas Zajac'


# Parses all SMILES of the input file
# Returns Python lists of IDs, SMILES and RDKit molecules
def parse_input(filename=str):
    orig_smiles = list()
    molecules = list()
    activity = list()
    ids = list()

    with open(filename, 'r') as f:
        for line in f:
            split = line.split('\t')
            smile = split[0].replace('"', '')
            mol = Chem.MolFromSmiles(smile)
            class_label = int(split[1].split('\n')[0].rstrip())
            if mol is not None:
                orig_smiles.append(smile.rstrip())
                molecules.append(Chem.MolFromSmiles(smile))
                if class_label == 1:
                    activity.append(class_label)
                else:
                    activity.append(0)
        for i in range(1, len(molecules) + 1):  # generate IDs
            ids.append(i)
    # to np array
    activity = np.asarray(activity)

    return ids, orig_smiles, molecules, activity


# Generate and train the predictor
def final_predictor(molecules_train, activity_train, molecules_test):
    # Default Classifier
    clf = XGBClassifier(random_state=1, n_jobs=-1)

    # Generate feature vector training (80%)
    feature_vector_for_train = generate_initial_feature_vector(molecules_train, activity_train)
    feature_vector_train = select_most_important_features(clf, feature_vector_for_train, 103, activity_train,
                                                          feature_vector_for_train)

    # Generate feature vector testing (20%)
    feature_vector_test = generate_initial_feature_vector(molecules_train, activity_train, molecules_test)
    feature_vector_test = select_most_important_features(clf, feature_vector_for_train.copy(), 103, activity_train,
                                                         feature_vector_test)

    final_clf = XGBClassifier(random_state=1, n_jobs=-1, subsample=0.75, reg_lambda=2.0, reg_alpha=0.0,
                              n_estimators=150, min_child_weight=0.5, max_depth=8, learning_rate=0.2,
                              gamma=0.5, colsample_bytree=1.0)

    final_clf.fit(feature_vector_train, activity_train)

    predictions = final_clf.predict(feature_vector_test)

    return predictions


# create morgan fingerprint bit feature vector
def feature_fingerprint_morgan(molecules, radius, bits):
    feature_vector = []

    for mol in molecules:
        # feature_vector.append(AllChem.GetMorganFingerprint(mol, 3))
        # radius 3 equals ecpf6 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4510302/
        string_bits = GetMorganFingerprintAsBitVect(mol, radius, nBits=bits).ToBitString()
        feature_vector.append(list(map(int, string_bits)))
    return feature_vector


# create atom pair fingerprint
def feature_fingerprint_atom_pair(molecules, bits):
    feature_vector = []

    for mol in molecules:
        string_bits = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=bits).ToBitString()
        feature_vector.append(list(map(int, string_bits)))
    return feature_vector


# Generate feature vector with all fingerprints
def generate_molecular_descriptors(molecules):
    names_descriptors = [x[0] for x in Descriptors._descList]
    my_desc_obj = MoleculeDescriptors.MolecularDescriptorCalculator(names_descriptors)
    feature_vector = [my_desc_obj.CalcDescriptors(x) for x in molecules]
    feature_vector = np.asarray(feature_vector)
    feature_vector[np.isnan(feature_vector)] = 0  # Replace NaN with 0
    return feature_vector


# generate the feature vector for the training and the testing data
def generate_initial_feature_vector(molecules_1, activity, molecules_2=None):
    if molecules_2 is None:
        datasets = [molecules_1]
    else:
        datasets = [molecules_1, molecules_2]

    all_feature_vectors = []

    for d in datasets:
        # Generate feature vector
        feature_vector = generate_molecular_descriptors(d)

        # ECFP6 feature vector
        ecfp8 = np.asarray(feature_fingerprint_morgan(d, 4, 4096))

        # linear fingerprints feature vector (Atom pair=
        lin_fingerprint = np.asarray(feature_fingerprint_atom_pair(d, 4096))

        # combination
        combined_feature_vector = []
        for x, y, z in zip(ecfp8, feature_vector, lin_fingerprint):
            combined_feature_vector.append(x.tolist() + y.tolist() + z.tolist())
        combined_feature_vector = np.asarray(combined_feature_vector)

        all_feature_vectors.append(combined_feature_vector)

    # Default Classifier
    clf = XGBClassifier(random_state=1, n_jobs=-1)

    clf.fit(all_feature_vectors[0], activity)
    importances = clf.feature_importances_
    boolean_mask = importances != 0

    if molecules_2 is None:
        new_feature_vector = all_feature_vectors[0][:, boolean_mask]
    else:
        new_feature_vector = all_feature_vectors[1][:, boolean_mask]

    return new_feature_vector


# reduce the number of features to the k most important features
def select_most_important_features(classifier, feature_vector_basis, number_features, activity, vector_to_reduce):
    classifier.fit(feature_vector_basis, activity)
    return vector_to_reduce[:, classifier.feature_importances_.argsort()[::-1][:number_features]]


# Calculates and prints scores
def calculate_scores(y_true, y_pred):
    print("MCC:\t" + str(matthews_corrcoef(y_true, y_pred)))
    print("ACC:\t" + str(accuracy_score(y_true, y_pred)))
    print("SE:\t" + str(recall_score(y_true, y_pred)))
    print("SP:\t" + str(my_specificity_score(y_true, y_pred)))
    print("ROCAUC:\t" + str(roc_auc_score(y_true, y_pred)))

    # specificity scoring function

def my_specificity_score(labels, predictions):
    tp = fp = tn = fn = 0
    for x in range(len(labels)):
        if (predictions[x] == labels[x]) and (labels[x] == 1):
            tp += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 1):
            fn += 1
        elif (predictions[x] == labels[x]) and (labels[x] == 0):
            tn += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 0):
            fp += 1
    score = tn / (tn + fp)
    return score




# Write Output with original labels and predicted labels
def write_output(orig_smiles_predict, labels_predict, labels_original, my_output):
    with open(my_output, 'w+') as f:
        for smiles, label, label_orig in zip(orig_smiles_predict, labels_predict, labels_original):
            if label == 0:
                label = -1
            if label_orig == 0:
                label_orig = -1
            f.write('\t'.join([smiles, str(label_orig), str(label)]) + '\n')


# Example command
# PredictorLemkeZajac.py -i data/training_data.csv -o output_Lemke_Zajac.csv

def main(argv):
    print('QSAR Project')
    print('Author: ' + __author__)

    help_string = "Command line example: \nPredictorLemkeZajac.py -i <input_prediction_data> " \
                  "-o <output_prediction_data>"
    my_input = ""
    my_output = ""
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["help", "input=", "output="])
        if opts == []:
            print(help_string)
            sys.exit()
    except getopt.GetoptError:
        print(help_string)
        sys.exit()
    for o, a in opts:
        if a == '' or o == '':
            print(help_string)
            sys.exit()
        elif o in ("-h", "--help"):
            print(help_string)
            sys.exit()
        elif o in ("-i", "--input"):
            my_input = a
        elif o in ("-o", "--output"):
            my_output = a

    # Parse the training data to generate a list of RDKit molecules
    ids_train, orig_smiles_train, molecules_train, activity_train = parse_input("data/training_data.csv")

    print("asdf")

    # Parse the prediction data to generate a list of RDKit molecules
    ids_predict, orig_smiles_predict, molecules_test, activity_test = parse_input(my_input)

    # train and predict
    predicted_class_labels = final_predictor(molecules_train, activity_train, molecules_test)

    # calculate and print scores
    calculate_scores(activity_test, predicted_class_labels)

    # Write the combined ids, smarts, number of atoms and bonds to output file
    write_output(orig_smiles_predict, predicted_class_labels, activity_test, my_output)


if __name__ == "__main__":
    main(sys.argv[1:])
