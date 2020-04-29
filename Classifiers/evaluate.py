"""Class to write output"""
import csv


def evaluate(classifier, index, y_pred):
    # Write output to csv file
    with open('result.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter=',',lineterminator = '\n')
        for i in range(0, len(index) + 1):
            row = [index[i], y_pred[i]]
            print(row)
            writer.writerow(row)
