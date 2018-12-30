# import all required libraries here
import sys
import getopt
from rdkit import Chem
import sklearn
import itertools
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas as pd

__author__ = 'Steffen Lemke'


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
        for i in range(1, len(molecules) + 1):
            ids.append(i)

    return ids, orig_smiles, molecules, activity


def generate_feature_vector(molecules):
    names_descriptors = [x[0] for x in Descriptors._descList]
    my_desc_obj = MoleculeDescriptors.MolecularDescriptorCalculator(names_descriptors)
    feature_vector = [my_desc_obj.CalcDescriptors(x) for x in molecules]
    return feature_vector


def my_cross_validation(classifier, feature_vector, labels):
    scoring = {'F1': 'f1',
               'Accuracy': 'accuracy',
               'ROC_AUC': 'roc_auc'}

    scores = cross_validate(classifier, feature_vector, labels, cv=10, scoring=scoring)
    # print("F1 Score: " + str(score.mean()) + " +- " + str(score.std()))
    # print(scores[test_])
    for key in scoring.keys():
        single_score_array = scores['test_' + key]
        print(key + ":  " + str(single_score_array.mean()) + " +- " + str(single_score_array.std()))


def write_output(ids, orig_smiles, cluster_labels, output):
    with open(output, 'w') as f:
        for id, smrt, label in zip(ids, orig_smiles, cluster_labels):
            f.write('\t'.join([str(id), smrt, str(label)]) + '\n')


def recursive_feature_elimination(classifier, feature_vector, labels):
    rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(3), scoring='f1', n_jobs=-1)
    rfecv.fit(feature_vector, labels)

    print("Optimal number of features : %d" % rfecv.n_features_)
    # print("Mask of selected features :")
    # print(rfecv.support_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    names_descriptors = np.asarray([x[0] for x in Descriptors._descList])
    important_features = np.asarray(names_descriptors)[rfecv.support_]
    # print(important_features)

    return rfecv.support_


# Example command
# PredictorLemkeZajac.py -i data/training_data.csv -o output_Lemke_Zajac.txt

def main(argv):
    print('QSAR Project')
    print('Author: ' + __author__)

    help_string = "Command line example: \nPredictorLemkeZajac.py -i <input_file> " \
                  "-o <output_file>"
    input = ""
    output = ""
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
            input = a
        elif o in ("-o", "--output"):
            output = a

    # Parse the input file to generate a list of RDKit molecules
    ids, orig_smiles, molecules, activity = parse_input(input)

    # Generate feature vector
    feature_vector = generate_feature_vector(molecules)
    feature_vector = np.asarray(feature_vector)
    feature_vector[np.isnan(feature_vector)] = 0

    activity = np.asarray(activity)
    # print(sorted(sklearn.metrics.SCORERS.keys()))

    # Classifier
    classifier = XGBClassifier(seed=1)


    my_cross_validation(classifier, feature_vector, activity)




    # Recursive feature elimination
    feature_mask = recursive_feature_elimination(classifier, feature_vector, activity)

    print(feature_mask)
    #feature_vector = feature_vector[:, np.asarray(feature_mask)]
    #print(feature_vector)




    # Write the combined ids, smarts, number of atoms and bonds to output file
    # write_output(ids, orig_smiles, cluster_labels, output)


if __name__ == "__main__":
    main(sys.argv[1:])
