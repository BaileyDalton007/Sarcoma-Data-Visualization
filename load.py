import csv
import datetime

def getFullData():
    data = []
    with open('SARC_LMS_256_10_val_imgnet_pred_fc3_features.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            data.append(row)

        for i in range(len(data)): # fixes first and last embed being a string instead of a float
            if i > 0:
                data[i][8] = float(data[i][8].replace("[", ""))
                data[i][263] = float(data[i][263].replace("]", ""))
    return data

def getEmbedData(confidence):
    data = getFullData()
    em_data = []
    for i in range(len(data)):
            if i > 0:
                if float(data[i][4]) >= confidence:
                    em_data.append(data[i][-256:])
            else:
                em_data.append(data[i][-256:])
    return em_data, data
