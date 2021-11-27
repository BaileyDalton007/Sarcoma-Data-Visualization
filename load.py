import pandas as pd

# input data file
file = 'SARC_LMS_256_10_val_histossl_pred_fc3_features.csv'

def numerize(item):
    return float(item.replace("[","").replace("]","")) # fixes first and last embeds being loaded as strings

def getData(confidence, group_size):
    # loading script written by Asmaa
    csv_df = pd.read_csv(file)
    file_name = file
    
    test_gt = csv_df.iloc[:, 1]
    test_pred_class = csv_df.iloc[:, 2]
    test_prob = csv_df.iloc[:, 3]

    count_0, count_1, count_2 = 0, 0, 0

    indices = []
    for i in range(len(test_prob)):
         if test_prob[i] >= confidence:
            if test_pred_class[i] == 0 and count_0 < group_size:
                count_0 += 1
                indices.append(i)
            elif test_pred_class[i] == 1 and count_1 < group_size:
                count_1 += 1
                indices.append(i)
            elif test_pred_class[i] == 2 and count_2 < group_size:
                count_2 += 1
                indices.append(i)

    print(f'count_0: {count_0}')
    print(f'count_1: {count_1}')
    print(f'count_2: {count_2}')

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
