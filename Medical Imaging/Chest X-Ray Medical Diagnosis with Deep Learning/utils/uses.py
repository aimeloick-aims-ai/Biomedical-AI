import numpy as np
import torch

def get_predictions(model, loader, device):

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            # logits -> probabilities
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_prob

def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """ 
    df1_patients_unique = df1[patient_col].unique()
    df2_patients_unique = df2[patient_col].unique()
    
    patients_in_both_groups = np.intersect1d(df1_patients_unique,df2_patients_unique)
    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) > 0  # boolean (true if there is at least 1 patient in both groups)
    return leakage