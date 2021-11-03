import csv

data = []
with open('SARC_LMS_256_10_val_imgnet_pred_fc3_features.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0 
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            data.append(row)

print(len(data[0]))