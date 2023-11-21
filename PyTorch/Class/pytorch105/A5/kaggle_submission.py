import os
import csv
import numpy as np


def write_csv(preds_submission):
    csv_file_path = './result.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['id', 'image_id', 'prediction_list'])

        for idx, (image_id, predictions) in enumerate(preds_submission.items(), start=1):
            csv_writer.writerow([idx, image_id, predictions])
