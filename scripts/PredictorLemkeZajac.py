# import all required libraries here
import sys
import getopt
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from xgboost import XGBClassifier


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


# Generate all descriptors available in the RDKit package
def generate_all_descriptors(molecules):
    names_descriptors = [x[0] for x in Descriptors._descList]
    my_desc_obj = MoleculeDescriptors.MolecularDescriptorCalculator(names_descriptors)
    feature_vector = [my_desc_obj.CalcDescriptors(x) for x in molecules]
    return feature_vector


# create morgan fingerprint bit feature vector
def feature_fingerprint_morgan(molecules, radius):
    feature_vector = []
    for mol in molecules:
        string_bits = GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToBitString()
        feature_vector.append(list(map(int, string_bits)))
    return feature_vector

# Generate Feature vector with Morgan Fingerprints and RDKit Descriptors
def generate_feature_vector(molecules):
    # Generate feature vector
    feature_vector = generate_all_descriptors(molecules)
    feature_vector = np.asarray(feature_vector)
    feature_vector[np.isnan(feature_vector)] = 0

    # ECFP6 feature vector
    ecfp6 = np.asarray(feature_fingerprint_morgan(molecules, 3))

    descriptors_ecfp6 = []
    for x, y in zip(ecfp6, feature_vector):
        descriptors_ecfp6.append(x.tolist() + y.tolist())

    descriptors_ecfp6 = np.asarray(descriptors_ecfp6)
    return descriptors_ecfp6


# Generate and train the predictor
def final_predictor(molecules, activity):
    # generate feature vector
    descriptors_ecfp6 = generate_feature_vector(molecules)

    final_clf = XGBClassifier(random_state=1, subsample=0.5, reg_lambda=0.5, reg_alpha=0.0, n_estimators=150, n_jobs=-1,
                              min_child_weight=0.5, max_depth=5, learning_rate=0.05, gamma=0.0, colsample_bytree=0.5)

    final_clf.fit(descriptors_ecfp6, activity)

    return final_clf

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

    # Parse the prediction data to generate a list of RDKit molecules
    ids_predict, orig_smiles_predict, molecules_predict, activity_predict = parse_input(my_input)

    # Generate and train predictor
    final_clf = final_predictor(molecules_train, activity_train)

    # Generate the feature vector for the prediction data
    feature_vector_predict = generate_feature_vector(molecules_predict)

    # Predict classes for the prediction data
    predicted_class_labels = final_clf.predict(feature_vector_predict)

    # Write the combined ids, smarts, number of atoms and bonds to output file
    write_output(orig_smiles_predict, predicted_class_labels, activity_predict, my_output)


if __name__ == "__main__":
    main(sys.argv[1:])
