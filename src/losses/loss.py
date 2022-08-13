import torch
import numpy as np


class HardNnetLoss(torch.nn.Module):
    """
    from Similarity_measures import mutual_info,SIFT_features
    """

    def __init__(self, anchorIds, positiveIds, negativeIds, lossweight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)
        # print("anchorIds" ,anchorIds )
        self.anchorIds = anchorIds
        self.positiveIds = positiveIds
        self.negativeIds = negativeIds
        self.lenpositive = len(self.positiveIds)
        self.lennegative = len(self.negativeIds)
        self.weight = lossweight
        self.margin = margin
        self.labelsMiniBatch = None
        self.sizeMiniBatch = len(anchorIds)
        assert len(anchorIds) == len(positiveIds) and len(positiveIds) == len(
            negativeIds
        )

    def getWeight(self):
        return self.weight

    def setActualLabels(self, newLabel):
        self.labelsMiniBatch = newLabel
        return

    def get_label_batch(label_data, batch_size, batch_index):

        nrof_examples = np.size(label_data, 0)
        j = batch_index * batch_size % nrof_examples

        if j + batch_size <= nrof_examples:
            batch = label_data[j : j + batch_size]
        else:
            x1 = label_data[j:nrof_examples]
            x2 = label_data[0 : nrof_examples - j]
            batch = np.vstack([x1, x2])
        batch_int = batch.astype(np.int64)

        return batch_int
