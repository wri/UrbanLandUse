import numpy as np



def calc_confusion(Yhat,Y,categories,silent=False):
    n_categories = len(categories)
    confusion = np.zeros((n_categories,n_categories),dtype='uint32')
    for j in range(n_categories):
        for i in range(n_categories):
            confusion[j,i] = np.sum(np.logical_and((Yhat==categories[i]),(Y==categories[j])))
            # print j,i, confusion[j,i], categories[j], categories[i] 
        print categories[j], np.sum(confusion[j,:])
    if not silent:
	    print confusion
	    print confusion.sum(), confusion.trace(), confusion.trace()/float(confusion.sum())
    return confusion


def calc_confusion_details(confusion):
    n_categories = confusion.shape[0]
    # out of samples in category, how many assigned to that category
    # (true positives) / (true positives + false negatives)
    # (correct) / (samples from category)
    recalls = np.zeros(n_categories, dtype='float32')
    # out of samples assigned to category, how many belong to that category
    # (true positives) / (true positives + false positives)
    # (correct) / (samples assigned to category)
    precisions = np.zeros(n_categories, dtype='float32')

    for j in range(n_categories):
        ascribed = np.sum(confusion[:,j])
        actual = np.sum(confusion[j,:])
        correct = confusion[j,j]
        recalls[j] = float(correct)/float(actual)
        precisions[j] = float(correct)/float(ascribed)

    # what percentage of total samples were assigned to the correct category
    accuracy = confusion.trace()/float(confusion.sum())

    return recalls, precisions, accuracy

