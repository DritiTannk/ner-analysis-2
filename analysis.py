import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, precision_recall_fscore_support

import matplotlib.pyplot as plt

if __name__ == "__main__":
    features = ['SR_NO', 'sentence_list', 'np_list', 'vp_list']

    main_ds = pd.read_csv('Data/NER_dataset.csv')

    main_ds = main_ds.dropna()  # Drop null rows.

    train_ds = main_ds[main_ds['label'] == 'TRAIN']

    # Extract features values for train rows
    train_x = train_ds[train_ds.columns[0:4]]

    # Extract target values for train rows
    train_y = train_ds['ner_rich']

    X = pd.get_dummies(train_x, columns=features, drop_first=True)

    # ------------------------- Test Data ---------------------

    test_ds = main_ds[main_ds['label'] == 'TEST']

    test_x = test_ds[test_ds.columns[0:4]]

    # Extract target values for test rows
    test_y = test_ds['ner_rich']

    X1 = pd.get_dummies(test_x, columns=features, drop_first=True)
    X1 = X1.reindex(columns=X.columns, fill_value=0)

    # ----------------------- LOGISTIC ALGORITHM ----------------

    log_algo = LogisticRegression()
    log_algo.fit(X, train_y)
    log_y_pred = log_algo.predict(X1)

    result = confusion_matrix(test_y, log_y_pred)
    print('\n\n result --> ', result)

    ConfusionMatrixDisplay(result, display_labels=log_algo.classes_).plot(
                                                                          cmap='Blues',
                                                                          xticks_rotation='horizontal',
                                                                          colorbar=False
                                                                        )

    precision_rate = precision_score(test_y, log_y_pred)
    print('\n\n PRECISION RATE ==> ', precision_rate)

    recall_rate = recall_score(test_y, log_y_pred)
    print('\n\n RECALL RATE ==> ', recall_rate)

    f1_rate = f1_score(test_y, log_y_pred)
    print('\n\n F1 SCORE  ==>  ', f1_rate)

    prf_rate = precision_recall_fscore_support(test_y, log_y_pred)
    print('\n\n PRF RATE ==> ', prf_rate)

    plt.savefig('Data/conf_matrix/logs_conf_mat.png')


# --------------- Naives Bayes --------------------------

    bayes = GaussianNB()
    bayes.fit(X, train_y)
    bayes_y_pred = bayes.predict(X1)

    print('\n\n --------- Bayes Y Predictation ---------- \n\n', bayes_y_pred)

    bayes_conf = confusion_matrix(test_y, bayes_y_pred, labels=[0, 1])
    print('\n\n result --> ', bayes_conf)

    ConfusionMatrixDisplay(bayes_conf, display_labels=bayes.classes_).plot(
                                                                            cmap='Blues',
                                                                            xticks_rotation='horizontal',
                                                                            colorbar=False
                                                                          )

    bayes_precision_rate = precision_score(test_y, bayes_y_pred)
    print('\n\n PRECISION RATE ==> ', bayes_precision_rate)

    bayes_recall_rate = recall_score(test_y, bayes_y_pred)
    print('\n\n RECALL RATE ==> ', bayes_recall_rate)

    bayes_f1_rate = f1_score(test_y, bayes_y_pred)
    print('\n\n F1 SCORE  ==>  ', bayes_f1_rate)

    bayes_prf_rate = precision_recall_fscore_support(test_y, bayes_y_pred)
    print('\n\n PRF RATE ==> ', bayes_prf_rate)

    plt.savefig('Data/conf_matrix/bayes_conf_matrix')

    # -------------------------- K-NN Algorithm ------------------------------

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, train_y)
    knn_y_pred = knn.predict(X1)

    print('\n\n --------------- KNN Y Prediction --------------- \n\n', knn_y_pred)

    knn_conf = confusion_matrix(test_y, knn_y_pred, labels=[0, 1])
    print('\n\n result --> ', knn_conf)

    ConfusionMatrixDisplay(knn_conf, display_labels=knn.classes_).plot(
                                                                        xticks_rotation='horizontal',
                                                                        colorbar=False)

    knn_precision_rate = precision_score(test_y, knn_y_pred)
    print('\n\n PRECISION RATE ==> ', knn_precision_rate)

    knn_recall_rate = recall_score(test_y, knn_y_pred)
    print('\n\n RECALL RATE ==> ', knn_recall_rate)

    knn_f1_rate = f1_score(test_y, knn_y_pred)
    print('\n\n F1 SCORE  ==>  ', knn_f1_rate)

    knn_prf_rate = precision_recall_fscore_support(test_y, knn_y_pred)
    print('\n\n PRF RATE ==> ', knn_prf_rate)

    plt.savefig('Data/conf_matrix/knn_conf_matrix')

    # ---------------------- Decision Tree Algorithm ------------------------

    dtree = DecisionTreeClassifier()
    dtree.fit(X, train_y)
    tree_y_pred = dtree.predict(X1)

    print('\n\n ------------------ Tree Y Prediction -------------------- \n\n ', tree_y_pred)

    tree_conf = confusion_matrix(test_y, tree_y_pred, labels=[0, 1])
    print('\n\n result --> ', tree_conf)

    ConfusionMatrixDisplay(tree_conf, display_labels=dtree.classes_).plot(
                                                                            cmap='RdPu',
                                                                            xticks_rotation='horizontal',
                                                                            colorbar=False
                                                                        )

    tree_precision_rate = precision_score(test_y, tree_y_pred)
    print('\n\n PRECISION RATE ==> ', tree_precision_rate)

    tree_recall_rate = recall_score(test_y, tree_y_pred)
    print('\n\n RECALL RATE ==> ', tree_recall_rate)

    tree_f1_rate = f1_score(test_y, tree_y_pred)
    print('\n\n F1 SCORE  ==>  ', tree_f1_rate)

    tree_prf_rate = precision_recall_fscore_support(test_y, tree_y_pred)
    print('\n\n PRF RATE ==> ', tree_prf_rate)

    plt.savefig('Data/conf_matrix/tree_conf_matrix.png')

    # ----------------------------- SVM Algorithm --------------------------------

    svm = SVC(kernel='linear')
    svm.fit(X, train_y)
    svm_y_pred = svm.predict(X1)

    print('\n\n -------------------- SVM Y Prediction -------------------\n\n', svm_y_pred)
    print('\n\n len of y predict data ==> ', len(svm_y_pred))

    svm_conf = confusion_matrix(test_y, svm_y_pred, labels=[0, 1])
    print('\n\n result --> ', svm_conf)

    ConfusionMatrixDisplay(svm_conf, display_labels=svm.classes_).plot(
                                                                        cmap='binary',
                                                                        xticks_rotation='horizontal',
                                                                        colorbar=False
                                                                      )

    svm_precision_rate = precision_score(test_y, svm_y_pred)
    print('\n\n PRECISION RATE ==> ', svm_precision_rate)

    svm_recall_rate = recall_score(test_y, svm_y_pred)
    print('\n\n RECALL RATE ==> ', svm_recall_rate)

    svm_f1_rate = f1_score(test_y, svm_y_pred)
    print('\n\n F1 SCORE  ==>  ', svm_f1_rate)

    svm_prf_rate = precision_recall_fscore_support(test_y, svm_y_pred)
    print('\n\n PRF RATE ==> ', svm_prf_rate)

    plt.savefig('Data/conf_matrix/svm_conf_matrix')



