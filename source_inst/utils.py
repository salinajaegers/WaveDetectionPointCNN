# import libraries
import numpy as np
import torch
import torch.nn.functional as F

#intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)


# Set the CPU/GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class F1MetricAccumulator():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.n_true_positives = 0
        self.n_false_positives = 0
        self.n_false_negatives = 0
        self.n_true_negatives = 0
        self.n_samples = 0
        self.sigmoid = torch.nn.Sigmoid()

    def update(self, pred_cls, pred_inst, inst):
        # Ensure inputs are on the correct device
        pred_cls = pred_cls.to(device)
        pred_cls = self.sigmoid(pred_cls)
        pred_inst = pred_inst.to(device)
        inst = inst.to(device)

        # Check sizes match
        assert pred_inst.size(0) == inst.size(0) == pred_cls.size(0)

        n_points = inst.size(0)

        # Initialize true positives and false positives
        true_positives = torch.zeros(n_points, dtype=torch.float).to(device)
        false_positives = torch.zeros(n_points, dtype=torch.float).to(device)
        false_negatives = torch.zeros(n_points, dtype=torch.float).to(device)
        true_negatives = torch.zeros(n_points, dtype=torch.float).to(device)

        # True Positives
        print('THE RANGE OF THE CLASSSSESSS PREDICTIONN')
        print(torch.max(pred_cls))
        print(torch.min(pred_cls))
        print('SUM OF THE -1 INSTANCES BC GOOOOD WTF')
        print(torch.sum((inst != -1)))
        print(torch.sum((inst == -1)))

        true_positives[(pred_cls >= self.threshold) & (inst != -1)] = 1

        # False Positives
        false_positives[(pred_cls >= self.threshold) & (inst == -1)] = 1

        # False Negatives
        false_negatives[(pred_cls <= self.threshold) & (inst != -1)] = 1

        true_negatives[(pred_cls <= self.threshold) & (inst == -1)] = 1

        print('TRUE AND FALSE NEGATIVES')
        print(torch.sum(true_positives).item())
        print(torch.sum(false_positives).item())
        print(torch.sum(false_negatives).item())
        print(torch.sum(true_negatives).item())
        # Update counters
        current_tp = torch.sum(true_positives).item()
        current_fp = torch.sum(false_positives).item()
        current_fn = torch.sum(false_negatives).item()
        current_tn = torch.sum(true_negatives).item()
        self.n_true_positives += current_tp
        self.n_false_positives += current_fp
        self.n_false_negatives += current_fn
        self.n_true_negatives += current_tn
        
        # Precision
        precision = current_tp / (
            current_tp + current_fp + 1e-10)

        # Recall (no false negatives in this setup)
        recall = current_tp / (current_tp + current_fn + 1e-10)

        # F1 Score
        f1_score = (precision * recall) / ((precision + recall) / 2 + 1e-10)
        print('COMPUTTTTTTE')
        print(f1_score)
        return f1_score 

    def compute(self):
        # Compute precision
        precision = self.n_true_positives / (
            self.n_true_positives + self.n_false_positives + 1e-10)

        # Recall (no false negatives in this setup)
        recall = self.n_true_positives / (self.n_true_positives + self.n_false_negatives + 1e-10)

        # F1 Score
        f1_score = (precision * recall) / ((precision + recall) / 2 + 1e-10)
        print('COMPUTTTTTTE')
        print(f1_score)
        return f1_score



class AccuracyAccumulator():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.n_true_positives = 0
        self.n_positives = 0
        self.n_negatives = 0
        self.n_true_negatives = 0
        self.n_samples = 0

    def update(self, pred_cls, pred_inst, inst):
        # Ensure inputs are on the correct device
        pred_cls = pred_cls.to(device)
        pred_inst = pred_inst.to(device)
        inst = inst.to(device)

        # Check sizes match
        assert pred_inst.size(0) == inst.size(0) == pred_cls.size(0)

        n_points = inst.size(0)

        # Initialize true positives and false positives
        true_positives = torch.zeros(n_points, dtype=torch.float).to(device)
        positives = torch.zeros(n_points, dtype=torch.float).to(device)
        negatives = torch.zeros(n_points, dtype=torch.float).to(device)
        true_negatives = torch.zeros(n_points, dtype=torch.float).to(device)

        # True Positives
        print('THE RANGE OF THE CLASSSSESSS PREDICTIONN')
        print(torch.max(pred_cls))
        print(torch.min(pred_cls))
        print('SUM OF THE -1 INSTANCES BC GOOOOD WTF')
        print(torch.sum((inst != -1)))
        print(torch.sum((inst == -1)))

        true_positives[(pred_cls > self.threshold) & (inst != -1)] = 1

        # False Positives
        positives[(pred_cls > self.threshold)] = 1

        # False Negatives
        negatives[(pred_cls <= self.threshold)] = 1

        true_negatives[(pred_cls <= self.threshold) & (inst == -1)] = 1



        print('TRUE AND FALSE NEGATIVES')
        print(torch.sum(true_positives).item())
        print(torch.sum(positives).item())
        print(torch.sum(negatives).item())
        print(torch.sum(true_negatives).item())
        # Update counters
        current_tp = torch.sum(true_positives).item()
        current_p = torch.sum(positives).item()
        current_n = torch.sum(negatives).item()
        current_tn = torch.sum(true_negatives).item()
        self.n_true_positives += current_tp
        self.n_positives += current_p
        self.n_negatives += current_n
        self.n_true_negatives += current_tn
        

        # Accuracy Score
        #TP + TN / P + N
        accuracy_score = (current_tp + current_tn) / (current_p + current_n + 1e-10)
        print('UPDATA ACC')
        print(accuracy_score)
        return accuracy_score 

    def compute(self):
        # Accuracy Score
        #TP + TN / P + N
        accuracy_score = (self.n_true_positives + self.n_true_negatives) / (self.n_positives + self.n_negatives + 1e-10)

        print('COMPUTE ACCC')
        print(accuracy_score)
        return accuracy_score

def even_intervals(nepochs, ninterval=1):
    """Divide a number of epochs into regular intervals."""
    out = list(np.linspace(0, nepochs, num=ninterval, endpoint=False, dtype=int))[1:]
    return out




def instance_loss(embeddings, labels, margin=0.5):
    """
    embeddings: [N, D] (embedding space)
    labels: [N] (instance labels)
    margin: Distance margin for different instance points to be pushed apart
    """
    unique_labels = labels.unique()

    pull_loss = 0
    push_loss = 0
    num_instances = 0

    for label in unique_labels:
        if label == -1:
            continue  # Ignore background or unannotated points

        # Get embeddings of points belonging to this instance
        instance_mask = (labels == label)
        instance_embeddings = embeddings[instance_mask]

        if instance_embeddings.size(0) < 2:
            continue  # Ignore instances with less than 2 points

        # Pull loss: Encourage instance points to be close to their centroid
        centroid = instance_embeddings.mean(dim=0)
        pull_loss += ((instance_embeddings - centroid) ** 2).mean()

        # Push loss: Compare instance centroids and push them apart
        for other_label in unique_labels:
            if other_label == label or other_label == -1:
                continue
            other_instance_mask = (labels == other_label)
            other_instance_embeddings = embeddings[other_instance_mask]
            if other_instance_embeddings.size(0) < 2:
                continue
            other_centroid = other_instance_embeddings.mean(dim=0)
            push_loss += F.relu(margin - torch.norm(centroid - other_centroid))

        num_instances += 1

    return pull_loss / num_instances + push_loss / (num_instances * (num_instances - 1))
