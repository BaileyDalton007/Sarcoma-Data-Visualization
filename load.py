import csv

data = [] # 264 columns, 256 features
em_data = [] # only embeds
with open('SARC_LMS_256_10_val_imgnet_pred_fc3_features.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0 
    for row in csv_reader:
        data.append(row)

for i in range(len(data)):
        em_data.append(data[i][-256:])

for i in range(len(em_data)):
    if i > 0:                            # exludes the column names
        for j in range(len(em_data[i])):
            em_data[i][j] = em_data[i][j].replace("[", "")
            em_data[i][j] = float(em_data[i][j].replace("]", ""))
