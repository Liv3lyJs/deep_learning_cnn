import sys
import torch.nn as nn
import torch

import numpy as np


class Metric:
    def get_name(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def accumulate(self, prediction, target):
        raise NotImplementedError()

    def get_value(self):
        raise NotImplementedError()


class AccuracyMetric(Metric):
    def __init__(self, class_probability_threshold):
        self.class_probability_threshold = class_probability_threshold

        self._good_count = 0.0
        self._total_count = 0.0

    def get_name(self):
        return 'Accuracy'

    def clear(self):
        self._good_count = 0.0
        self._total_count = 0.0

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)

        :param prediction: Le tenseur PyTorch de prédiction des classes des images
            Dimensions : (N, 3)
                (i, 0) : probabilité qu'un cercle est présent dans l'image i
                (i, 1) : probabilité qu'un triangle est présent dans l'image i
                (i, 2) : probabilité qu'une croix est présente dans l'image i
        :param target: Le tenseur PyTorch cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        prediction = (prediction > self.class_probability_threshold).astype(float)
        self._good_count += (prediction == target).sum()
        self._total_count += prediction.size

    def get_value(self):
        return self._good_count / self._total_count


class MeanAveragePrecisionMetric(Metric):
    def __init__(self, class_count, intersection_over_union_threshold):
        self._class_count = class_count
        self._intersection_over_union_threshold = intersection_over_union_threshold

        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def get_name(self):
        return 'mAP'

    def clear(self):
        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)
        M: Nombre de prédictions fait par le modèle

        :param prediction:
            Dimensions : (N, M, 7)
                (i, j, 0) indique le niveau de confiance entre 0 et 1 qu'un vrai objet est représenté par le vecteur (n, m, :)
                Si (i, j, 0) est plus grand qu'un seuil :
                    (i, j, 1) est la position x centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 2) est la position y centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet prédit j de l'image i
                    (i, j, 4) est le score pour la classe "cercle" de l'objet prédit j de l'image i
                    (i, j, 5) est le score pour la classe "triangle" de l'objet prédit j de l'image i
                    (i, j, 6) est le score pour la classe "croix" de l'objet prédit j de l'image i

        :param target: Le tenseur cible pour la tâche de détection:
            Dimensions : (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        N = prediction.shape[0]
        C = prediction.shape[1]

        for n in range(N):
            for c in range(C):
                if target[n, c, 0] == 1:
                    class_index = int(target[n, c, 4])
                    self._target_count_by_class[class_index] += 1

            found_target = set()
            for c in range(C):
                confidence = prediction[n, c, 0]

                iou, target_index = self._find_best_target(prediction[n, c], target[n])
                target_class = int(target[n, target_index, 4])
                predicted_class = np.argmax(prediction[n, c, 4:])

                true_positive = 0
                false_positive = 0
                if target_index in found_target or iou < self._intersection_over_union_threshold:
                    false_positive = 1
                elif iou > self._intersection_over_union_threshold and predicted_class == target_class:
                    true_positive = 1
                found_target.add(target_index)

                self._results_by_class[predicted_class].append({
                    'confidence': confidence,
                    'true_positive': true_positive,
                    'false_positive': false_positive,
                })

    def _find_best_target(self, prediction, target):
        C = target.shape[0]
        ious = np.zeros(C)
        for c in range(C):
            if target[c, 0] == 1:
                iou = detection_intersection_over_union(prediction[1:4], target[c, 1:4])
                ious[c] = iou
            else:
                ious[c] = 0

        target_index = np.argmax(ious)
        return ious[target_index], target_index

    def get_value(self):
        mean_average_precision = 0
        for class_index in range(self._class_count):
            mean_average_precision += self._calculate_average_precision(self._results_by_class[class_index],
                                                                        self._target_count_by_class[class_index])

        return mean_average_precision / self._class_count

    def _calculate_average_precision(self, results, target_count):
        sorted_results = sorted(results, key=lambda result: result['confidence'], reverse=True)

        recalls = [0]
        precisions = [1]

        true_positive = 0
        false_positive = 0
        for result in sorted_results:
            true_positive += result['true_positive']
            false_positive += result['false_positive']

            recalls.append(true_positive / target_count if target_count > 0 else 0)

            precision_denominator = true_positive + false_positive
            precisions.append(true_positive / precision_denominator if precision_denominator > 0 else 1)

        recalls = np.array(recalls)
        precisions = np.array(precisions)

        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]

        return np.trapz(y=precisions, x=recalls)


def construct_best_boxes_tensor(predictions, labels, batch_size, iou_threshold=0.5):
    # Initialize the output tensor with zeros
    best_boxes_tensor = torch.zeros((batch_size, 3, 7))

    for i in range(batch_size):
        img_preds = predictions[i]  # Shape (7, 7, 3, 7)
        img_labels = labels[i]  # Shape (3, 5)

        # Counter to track how many objects are found in the image
        object_count = 0

        for label in img_labels:
            if label[0] == 0:  # No object in this row of labels
                continue

            # Get the ground truth box information
            gt_x, gt_y, gt_w, gt_class = label[1], label[2], label[3], label[4]
            ground_truth_box = (gt_x, gt_y, gt_w)

            best_iou = 0
            best_box = None

            for grid_x in range(7):
                for grid_y in range(7):
                    for b in range(3):
                        obj_prob = img_preds[grid_x, grid_y, b, 0]
                        if obj_prob < iou_threshold:
                            continue

                        # Extract the predicted box information
                        pred_x = img_preds[grid_x, grid_y, b, 1]
                        pred_y = img_preds[grid_x, grid_y, b, 2]
                        pred_w = img_preds[grid_x, grid_y, b, 3]
                        pred_class_logits = img_preds[grid_x, grid_y, b, 4:]
                        pred_class = torch.argmax(pred_class_logits).item()

                        predicted_box = (pred_x, pred_y, pred_w)

                        # Compute IoU
                        iou = detection_intersection_over_union(predicted_box, ground_truth_box)

                        # Update best match if this box has a higher IoU and matches the class
                        if iou > best_iou and pred_class == gt_class:
                            best_iou = iou
                            best_box = [obj_prob.item(), pred_x.item(), pred_y.item(), pred_w.item(), pred_class, iou,
                                        gt_class]

            # Store the best match in the tensor if found
            if best_box and object_count < 3:
                best_boxes_tensor[i, object_count] = torch.tensor(best_box)
                object_count += 1

    return best_boxes_tensor


def find_best_boxes(predictions, labels, iou_threshold=0.5):
    # Initialize a list to store the best matches for each image in the batch
    best_matches = []

    batch_size = predictions.shape[0]
    for i in range(batch_size):
        img_preds = predictions[i]  # Shape (7, 7, 3, 7)
        img_labels = labels[i]  # Shape (3, 5)

        # List to store the best boxes for this image
        img_best_boxes = []

        for label in img_labels:
            if label[0] == 0:  # No object in this label
                continue

            # Get the ground truth box information
            gt_x, gt_y, gt_w, gt_class = label[1], label[2], label[3], label[4]
            ground_truth_box = (gt_x, gt_y, gt_w)

            best_iou = 0
            best_box = None

            for grid_x in range(7):
                for grid_y in range(7):
                    for b in range(3):
                        obj_prob = img_preds[grid_x, grid_y, b, 0]

                        # Check object probability threshold
                        if obj_prob < 0.5:
                            continue

                        # Extract the predicted bounding box values
                        pred_x = img_preds[grid_x, grid_y, b, 1]
                        pred_y = img_preds[grid_x, grid_y, b, 2]
                        pred_w = img_preds[grid_x, grid_y, b, 3]
                        pred_class_logits = img_preds[grid_x, grid_y, b, 4:]

                        # Create the predicted box in the same format as ground truth
                        predicted_box = (pred_x, pred_y, pred_w)

                        # Compute IoU between the predicted box and ground truth box
                        iou = detection_intersection_over_union(predicted_box, ground_truth_box)

                        # Update the best box if IoU is higher and meets probability criteria
                        if iou > best_iou:
                            best_iou = iou
                            best_box = img_preds[grid_x, grid_y, b]  # Store the entire 7 values

            # Store the best match in the list for this image if found
            if best_box is not None:
                img_best_boxes.append(best_box)

        # Pad with zeros if fewer than 3 boxes were found
        while len(img_best_boxes) < 3:
            img_best_boxes.append(torch.zeros(7))

        # Only take the first 3 best matches if more than 3 were found
        best_matches.append(torch.stack(img_best_boxes[:3]))

    # Convert list of tensors to a single tensor of shape (N, 3, 7)
    return torch.stack(best_matches)
def alexnet_loss2(predictions, targets):  # predictions: Nx3x7, targets: Nx3x5
    A = 1
    B = 1
    batch_size = predictions.shape[0]
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    obj_count = 0
    # Step 1: Iterate over each image in the batch
    for i in range(batch_size):
        target = targets[i]  # Shape: (3, 5)
        prediction = predictions[i]  # Shape: (3, 7)
        box_loss = 0.0
        class_loss = 0.0

        # Step 2: Iterate over each target object in the image
        for j in range(3):
            presence = target[j, 0]  # Step 3: Extract presence indicator (0 or 1)

            # Step 3: If the object is not present, skip to the next object
            if presence == 0:
                continue
            obj_count += 1
            # Extract target box coordinates and class label
            target_box = target[j, 1:4]  # Coordinates: (x, y, size)
            class_label = target[j, 4].long()  # Class label

            best_iou = -1
            best_pred_idx = -1

            # Step 4: Find the prediction box with the highest IoU for the current target box
            for k in range(3):
                pred_box = prediction[k, 1:4]  # Predicted box coordinates (x, y, size)
                iou = detection_intersection_over_union(pred_box, target_box)

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = k

            # Step 5: Calculate the IoU loss for the best matching prediction box
            box_loss += 1 - best_iou  # Loss is 1 - IoU

            # Step 6: Calculate the class loss for the best matching prediction
            # Extract the class logits from the last 3 columns for the best prediction
            best_pred_class_logits = prediction[best_pred_idx, 4:]  # Shape: (3,) - Logits for classes 0, 1, 2

            # Calculate cross-entropy loss for class prediction
            class_loss += ce_loss(best_pred_class_logits, class_label)

        # Combine the losses for this image
        total_loss += A * box_loss + B * class_loss

    # Average the loss over the batch
    total_loss = total_loss / obj_count

    return total_loss

def alexnet_loss(predictions, targets, classe):
    A = 1
    B = 1
    batch_size = predictions.shape[0]
    for i in range(batch_size):
        target = targets[i] #3x5
        prediction = predictions[i]
        currentline = None
        last_zero_line = None
        #check si prediction
        for j, object in target:
            if object[-1] == classe and target[0]==1:
                currentline = j
            if target[0] == 0:
                last_zero_line = j
        if currentline == None:
            currentline = last_zero_line

        L_xywh = (target[currentline][1] - prediction[1]) ** 2 + (target[currentline][2] - prediction[2]) ** 2 + 2 * ((np.sqrt(target[currentline][3]) - np.sqrt(prediction[3])) ** 2)
        L_class = nn.BCELoss()(prediction[0], target[currentline][0])
        L = A * L_xywh + B * L_class
        return L







def detection_intersection_over_union(box_a, box_b):
    area_a = box_a[2] ** 2
    area_b = box_b[2] ** 2

    a_tl_x = box_a[0] - box_a[2] / 2
    a_tl_y = box_a[1] - box_a[2] / 2
    a_br_x = box_a[0] + box_a[2] / 2
    a_br_y = box_a[1] + box_a[2] / 2

    b_tl_x = box_b[0] - box_b[2] / 2
    b_tl_y = box_b[1] - box_b[2] / 2
    b_br_x = box_b[0] + box_b[2] / 2
    b_br_y = box_b[1] + box_b[2] / 2

    intersection_w = min(a_br_x, b_br_x) - max(a_tl_x, b_tl_x)
    intersection_h = min(a_br_y, b_br_y) - max(a_tl_y, b_tl_y)
    intersection_w = max(intersection_w, 0)
    intersection_h = max(intersection_h, 0)

    intersection_area = intersection_w * intersection_h

    return intersection_area / (area_a + area_b - intersection_area)


class SegmentationIntersectionOverUnionMetric(Metric):
    def __init__(self, background_class):
        self._background_class = background_class

        self._intersection = 0.0
        self._union = 0.0
        self._epsilon = sys.float_info.epsilon

    def get_name(self):
        return 'Intersection Over Union'

    def clear(self):
        self._intersection = 0.0
        self._union = 0.0

    def accumulate(self, prediction, target):
        """
        Méthode qui accumule les métriques d'un lot de données.
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param prediction: Le tenseur PyTorch de prédiction des classes de chaque pixel des images
            Dimensions : (N, C, H, W) où C est le nombre de classes en incluant l'arrière-plan (4 dans la problématique)
            (n, c, h, w) : le score de la classe c du pixel (h,w) de l'image n

        :param target: Le tenseur PyTorch cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        """
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        intersection, union = segmentation_intersection_over_union(prediction, target, self._background_class)
        self._intersection += intersection
        self._union += union

    def get_value(self):
        return (self._intersection + self._epsilon) / (self._union + self._epsilon)


def segmentation_intersection_over_union(prediction, target, background_class):
    C = prediction.shape[1]
    intersection = 0.0
    union = 0.0
    predicted_segmentation = np.argmax(prediction, axis=1)

    for c in range(C):
        if c == background_class:
            continue

        predicted_mask = predicted_segmentation == c
        target_mask = target == c

        intersection += (predicted_mask * target_mask).sum()
        union += predicted_mask.sum() + target_mask.sum() - (predicted_mask * target_mask).sum()

    return intersection, union


class Custom_loss_detection():
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        for i, (entree_batch, cible_batch) in enumerate(zip(input, target)):  # Entree = 7, 7, 15  -  Cible = 3, 5
            print(f'Currently analysing batch: {i+1}')
            value_iou_obj1 = []
            # Si aucun objets
            if cible_batch[0, 0] == 1:
                # Itterer sur les rows de la batch
                for j in entree_batch: # Entree = 7, 15
                    # Itterer sur les columns de la batch
                    for k in j: # Entree = 15
                        # Envoyer les donnees de x, y, largeur de entree et cible 
                        value_iou_obj1.append(detection_intersection_over_union(box_a=k[1:4], box_b=cible_batch[0, 1:4]))
            else:
                print(f'There are no object in the batch')
                value_iou_obj1.append(0.0)

                    

            # for j, entree, cible in enumerate(zip(input, target), start=0):    
            #     print(f'We are analyzing data: {j+1}')
            #     print(f'Analyzing the first object...')
            #     value_iou = []
            #     if entree[:, :, 0] == 1 and entree[:, :, -1] == 0:
            #         box_entree = entree[:, :, 3]
            #         box_cible = cible[:, :, 3]
            #         value_iou.append(detection_intersection_over_union(box_entree, box_cible))
            #     else:
            #         print(f'There is no object in the first section')