import numpy as np

def iou(box1, box2):
    """Compute IoU between two boxes.
    Each box is defined by [xmin, ymin, xmax, ymax].
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def precision_recall_curve(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Compute precision and recall for a single image."""
    if len(pred_boxes) == 0:
        return np.array([0]), np.array([0])
    
    try : pred_boxes = sorted(pred_boxes, key=lambda x: x[-1], reverse=True)  # sort by confidence score
    except Exception as E :
        #print (pred_boxes)
        """for i in range(len(pred_boxes)):
            print (len(pred_boxes[i]),"2takir")
            for j in range(len(pred_boxes[i])):
                print (len(pred_boxes[i][j]),"3takir",(pred_boxes[i][j]))"""
                #pred_boxes[i][j]
        exit()
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = []

    for i, pred in enumerate(pred_boxes):
        pred_cls_id, pxmin, pymin, pxmax, pymax, pbox_id = pred
        best_iou = 0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            gt_cls_id, gxmin, gymin, gxmax, gymax, gbox_id = gt

            if pred_cls_id == gt_cls_id and j not in matched_gt:
                iou_value = iou([pxmin, pymin, pxmax, pymax], [gxmin, gymin, gxmax, gymax])

                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt_idx = j

        if best_iou > iou_threshold:
            tp[i] = 1
            matched_gt.append(best_gt_idx)
        else:
            fp[i] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
    recall = cumulative_tp / len(gt_boxes)

    return precision, recall

def compute_average_precision(precision, recall):
    """Compute the Average Precision (AP) from precision and recall."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap

def mean_average_precision(gt_list, pred_list, iou_threshold=0.5,batch_size=8):
    """Compute the Mean Average Precision (mAP) for a dataset."""
    means=[]
    aps = []
    
    for i in range(batch_size):#for gt_boxes, pred_boxes in zip(gt_list, pred_list):
        #print (len(gt_boxes),len(pred_boxes),"lennn")
        gt_boxes=gt_list[i]
        pred_boxes=pred_list[i]
        """print (len(pred_boxes),"DOUBLE KIR")
        for i in range(len(pred_boxes)):
            print (len(pred_boxes[i]),"2takir")
            for j in range(len(pred_boxes[i])):
                print (len(pred_boxes[i][j]),"3takir",(pred_boxes[i][j]))
                #pred_boxes[i][j]"""
        #exit()
        #print (gt_boxes)
        precision, recall = precision_recall_curve(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        ap = compute_average_precision(precision, recall)
        aps.append(ap)

    return np.mean(aps)
if __name__=="__main__":
# Example usage
    ground_truths = [
        [[1, 50, 50, 150, 150, 1], [2, 30, 30, 100, 100, 2]],  # ground truths for the first image
        [[1, 60, 60, 160, 160, 1]]  # ground truths for the second image
    ]

    predictions = [
        [[1, 55, 55, 155, 155, 0.9], [2, 25, 25, 105, 105, 0.8]],  # predictions for the first image
        [[1, 65, 65, 165, 165, 0.7]]  # predictions for the second image
    ]

    map_value = mean_average_precision(ground_truths, predictions)
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
