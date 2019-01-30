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

def extract_scoring_arrays(Yhat_img, Y_img, categories,
            remapping=None):
    # validate inputs
    assert Y_img.ndim==2
    # assert Y_img.dtype=='uint8'
    assert Y_img.dtype==Yhat_img.dtype
    assert Y_img.shape==Yhat_img.shape
    # remap ground-truth (Y) to match output (Yhat) categories
    if remapping is not None:
        if isinstance(remapping, str):
            if remapping.lower() == '3cat' or remapping.lower() == '3category':
                remapping = {2:2,3:2,4:2,5:2}
            elif remapping.lower() == 'roads':
                remapping = {0:0,1:0,2:0,3:0,4:0,5:0,6:1}
            else:
                raise ValueError('Unrecognized remapping identifier: ',remapping)
        assert isinstance(remapping, dict)
        for k in sorted(remapping.iterkeys()):
            Y_img[Y_img==k]=remapping[k]
    # create mask for presence of ground-truth (can include/exclude certain values if desired)
    mask = np.zeros(Y_img.shape, dtype='uint8')
    for c in categories:
        mask |= (Y_img == c)
    # identify and remove padding pixels
    nonpadding = (Yhat_img!=254)
    mask &= nonpadding
    # convert mask into series of locations (ie coordinates)
    locs = np.where(mask)
    # use coordinates to pull values from input images into arrays
    Yhat = Yhat_img[locs]
    Y = Y_img[locs]
    return Yhat, Y