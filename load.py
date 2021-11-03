import pandas as pd

# input data file
file = 'SARC_LMS_256_10_val_imgnet_pred_fc6_features.csv'

def numerize(item):
    return float(item.replace("[","").replace("]","")) # fixes first and last embeds being loaded as strings

def getData(confidence):
    # loading script written by Asmaa
    csv_df = pd.read_csv(file)
    file_name = file
    
    test_prob = csv_df.iloc[:, 3]

    indices = []
    for i in range(len(test_prob)):
         if test_prob[i] >= confidence:
             indices.append(i)

    pred_prob = csv_df.iloc[indices, 3].reset_index(drop=True)
    images = csv_df.iloc[indices, 0].reset_index(drop=True)
    gt = csv_df.iloc[indices, 1].reset_index(drop=True)
    pred_class = csv_df.iloc[indices, 2].reset_index(drop=True)

    n_classes = 3
    probs = csv_df.iloc[indices, 4:(4 + n_classes)].reset_index(drop=True)
    features = csv_df.iloc[indices, 4 + n_classes:].reset_index(drop=True)

    columns = []
    for col in features.columns:
        columns.append(col)

    features.iloc[:, 0] = features.iloc[:, 0].apply(numerize)
    features.iloc[:, len(columns)-1] = features.iloc[:, len(columns)-1].apply(numerize)

    return images, gt, pred_class, pred_prob, probs, features, columns, file_name

getData(0.9)