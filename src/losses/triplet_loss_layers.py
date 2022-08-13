import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


LOSS_LAYER_IDS = ["HardNetLoss", "AdaptativeTripletLoss"]


def loss_factory(loss_id, batch_size, margin_value, loss_weight):
    assert loss_id == "HardNetLoss"
    if loss_id == "HardNetLoss":
        loss_layer = HardNetLoss(
            batch_size=batch_size,
            margin=margin_value,
            loss_weight=loss_weight,
        )

    return loss_layer


class HardNetLoss(torch.nn.Module):
    def __init__(self, batch_size, loss_weight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)

        # construct anchor, positive, and negative ids
        self.anchor_ids = range(0, batch_size * 3, 3)
        self.positive_ids = range(1, batch_size * 3, 3)
        self.negative_ids = range(2, batch_size * 3, 3)

        self.nb_anchor_samples = len(self.anchor_ids)
        self.nb_positive_samples = len(self.positive_ids)
        self.nb_negative_samples = len(self.negative_ids)
        self.weight = loss_weight
        self.margin = margin
        self.labels_mini_batch = None
        self.sizeMiniBatch = len(self.anchor_ids)

        assert (
            self.nb_anchor_samples == self.nb_positive_samples
            and self.nb_positive_samples == self.nb_negative_samples
        )

    def get_weight(self):
        return self.weight

    def set_actual_labels(self, new_label):
        self.labels_mini_batch = new_label
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

    def forward(self, input):
        anchors = input[self.anchor_ids, :].cuda()
        positives = input[self.positive_ids, :].cuda()

        pos, min_neg, loss = loss_HardNet(
            anchors,
            positives,
            margin=self.margin,
            batch_reduce="min",
            loss_type="triplet_margin",
        )
        return pos, min_neg, loss


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1).cuda()
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1).cuda()

    eps = 1e-6
    return torch.sqrt(
        (
            d1_sq.repeat(1, positive.size(0))
            + torch.t(d2_sq.repeat(1, anchor.size(0)))
            - 2.0
            * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)
        )
        + eps
    ).cuda()


def loss_HardNet(
    anchor, positive, margin=1.0, batch_reduce="min", loss_type="triplet_margin"
):
    """
    HardNet margin loss - calculates loss based on distance matrix
    based on positive distance and closest negative distance.
    """

    assert (
        anchor.size() == positive.size()
    ), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    min_neg = torch.min(dist_without_min_on_diag, 1)[0]

    dist_matrix_a = distance_matrix_vector(anchor, anchor) + eps
    dist_matrix_p = distance_matrix_vector(positive, positive) + eps
    dist_without_min_on_diag_a = dist_matrix_a + eye * 10
    dist_without_min_on_diag_p = dist_matrix_p + eye * 10
    min_neg_a = torch.min(dist_without_min_on_diag_a, 1)[0]
    min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p, 0)[0])
    min_neg_3 = torch.min(min_neg_p, min_neg_a)
    min_neg = torch.min(min_neg, min_neg_3)

    min_neg = min_neg
    pos = pos1

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == "softmax":
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = -torch.log(exp_pos / exp_den)
    elif loss_type == "contrastive":
        loss = torch.clamp(margin - min_neg, min=0.0) + pos

    loss = torch.mean(loss)
    return pos, min_neg, loss


def dist_triplet(self, input):
    def distance_pair(x1, x2, norm=2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, norm).sum(dim=1)
        return torch.pow(out + eps, 1.0 / norm)

    anchors = input[self.anchorIds, :]
    positives = input[self.positiveIds, :]
    negatives = input[self.negativeIds, :]
    x1 = anchors
    x2 = positives
    x3 = negatives
    dist_positive = distance_pair(x1, x2, 2)
    dist_negative = distance_pair(x1, x3, 2)
    return (dist_positive, dist_negative)


class AdaptativeTripletLoss(torch.nn.Module):
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

    def forward(self, input):
        margin_factor = 1
        filter_loss = False
        categories = ["all", "easy", "semihard", "hard"]
        triplet_categorie = categories[0]
        loss = Variable(torch.zeros(1))
        zero = Variable(torch.zeros(1))
        anchors = input[self.anchorIds, :]
        positives = input[self.positiveIds, :]
        negatives = input[self.negativeIds, :]

        d_p = F.pairwise_distance(anchors, positives, 2)
        d_n = F.pairwise_distance(anchors, negatives, 2)

        self.margin = margin_factor * d_p
        self.margin = abs(d_p + d_n) / 2
        easy_samples = (d_p + self.margin < d_n).cpu().data.numpy().flatten()
        semihard_samples = (
            ((d_p < d_n) & (d_n < d_p + self.margin)).cpu().data.numpy().flatten()
        )
        hard_samples = (d_p > d_n).cpu().data.numpy().flatten()
        if triplet_categorie == "all":
            all = (d_p == d_p).cpu().data.numpy().flatten()
        if triplet_categorie == "easy":
            all = easy_samples
            # without margin
            # all = (d_p <  d_n) .cpu().data.numpy().flatten()
        if triplet_categorie == "semihard":
            # Semihard and d_n-d_p > 0|| d_p < d_n < d_p+margin
            # all2=np.where(np.logical_and(d_n < d_p + self.margin, d_n-d_p > 0> 0))[0]
            all = semihard_samples
        if triplet_categorie == "hard":
            #  Hard Triplet d_n<d_p
            all = hard_samples

        selected_triplets_ind = np.where(all == 1)
        if len(selected_triplets_ind[0]) == 0:
            return 0, [], []
        else:
            occ_selected_triplet = len(selected_triplets_ind[0])

        # Selected Triplet Data , output and labels..
        all_a = anchors[selected_triplets_ind, :]
        all_p = positives[selected_triplets_ind, :]
        all_n = negatives[selected_triplets_ind, :]
        aa = all_a.view(occ_selected_triplet, -1)
        pp = all_p.view(occ_selected_triplet, -1)
        nn = all_n.view(occ_selected_triplet, -1)

        dist_p = F.pairwise_distance(aa, pp, 2)
        dist_n = F.pairwise_distance(aa, nn, 2)

        # change margin
        self.margin = abs(dist_p + dist_n) / 2

        if triplet_categorie == "easy":
            loss2 = torch.max(zero, (dist_n - self.margin) - dist_p)  # +(d_aa-d_pp)
        else:
            loss2 = torch.max(
                zero, dist_p + self.margin - dist_n
            )  # +(d_aa-d_pp) # + 0.00001 * loss_embedd

        # Filter Valid loss!=0
        if filter_loss:
            all = (loss2 != 0).cpu().data.numpy().flatten()
            valid_loss = np.where(all == 1)[0]
            valid_losses = [loss2[u] for u in valid_loss]
            loss = np.mean(valid_losses)
        else:
            loss = torch.mean(loss2)

        return dist_p, dist_n, loss
