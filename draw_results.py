from csv import DictWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
import seaborn as sns

def calculate_cmc(y_test, probs, threshold=0.0):
    sorted_indices = np.argsort(-probs, axis=1)

    # Initialize ranks array with a value out of the normal rank range (e.g., n+1 for an 8 class problem)
    ranks = np.full(y_test.shape, len(np.unique(y_test))+1)

    for i in range(len(np.unique(y_test))):
        # Find indices where the true label matches the sorted index and the probability is above the threshold
        mask = (sorted_indices[:, i] == y_test) & (probs[np.arange(probs.shape[0]), sorted_indices[:, i]] >= threshold)
        ranks[mask] = i

    # Compute CMC curve
    cmc_counts = np.bincount(ranks, minlength=9)[:len(np.unique(y_test))]
    return np.cumsum(cmc_counts) / len(y_test)


def plot_cmc(cmc_curve, plot_title="Cumulative Match Characteristic (CMC) Curve", file_path=None, figsize=(10,6)):
    # Plotting the CMC curve
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=figsize)

    plt.plot(cmc_curve, marker='o', color='royalblue', linestyle='-', markersize=4)
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Recognition Rate", fontsize=14)
    plt.xticks(np.arange(0, len(cmc_curve), 1))  # Adjust according to your dataset's number of classes
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
    # Show plot
    plt.show()

# draw ROC curve
def draw_classes_roc_curve(y_test, y_score, classes, plot_title, file_path=None, print_classes=True,
                           print_micro_macro=True):
    lw = 2
    n_classes = len(classes)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    plt.clf()
    plt.figure(figsize=(8, 6))
    if print_micro_macro:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
    # Compute micro-average ROC curve and ROC area

    # Plot all ROC curves

    if print_classes:
        colors = cycle([
            "aqua", "midnightblue", "darkorange", "black", "slategray",
            "lightpink", "limegreen", "orchid", "aliceblue", "antiquewhite",
            "aqua", "aquamarine", "azure", "beige", "bisque", "blanchedalmond",
            "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
            "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson",
            "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray",
            "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen",
            "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
            "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise",
            "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
            "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia",
            "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green",
            "greenyellow", "grey", "honeydew", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
            "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
            "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
            "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
            "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow",
            "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
            "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
            "mediumslateblue", "mediumspringgreen", "mediumturquoise",
            "mediumvioletred", "midnightblue", "mintcream", "mistyrose",
            "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
            "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
            "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
            "pink", "plum", "powderblue", "purple", "red", "rosybrown",
            "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen",
            "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray",
            "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
            "thistle", "tomato", "turquoise", "violet", "wheat", "white",
            "whitesmoke", "yellow", "yellowgreen"
        ])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of user {0} (area = {1:0.2f})".format(
                    i, roc_auc[i]

                    # [name for name, id in classes.items() if id == i][0], roc_auc[i]),
                    # label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                ))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontdict={'size': 8})
    plt.ylabel("True Positive Rate", fontdict={'size': 8})
    plt.title(plot_title, fontdict={'size': 10})
    plt.legend(loc="lower right", fontsize=8)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()

# plot validation and accuracy through the epochs
def plot_result_nn(history, file_path=None):
    fig, axs = plt.subplots(2, 1)
    x = [k + 1 for k in range(len(history.history["loss"]))]
    # axs[0].plot(x, history.history["val_loss"], color="lightseagreen", label="Validation data loss", marker='.')
    axs[0].plot(x, history.history["loss"], color="fuchsia", label="Train data loss", marker='.')
    axs[0].set_title('Train and validation data loss over epochs.', fontsize=10)
    axs[0].set_ylabel('Data loss', fontsize=8)
    axs[0].legend(loc="upper right", fontsize=8)
    # axs[1].plot(x, history.history["val_accuracy"], "lightseagreen", label="Validation accuracy", marker='.')
    axs[1].plot(x, history.history["accuracy"], "fuchsia", label="Train accuracy", marker='.')
    axs[1].set_title('Train and validation accuracy over epochs.', fontsize=10)
    axs[1].set_ylabel('Accuracy', fontsize=8)
    axs[1].legend(loc="lower right", fontsize=8)

    for ax in axs.flat:
        ax.set(xlabel='Epochs')
        ax.label_outer()
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()


def get_info_readme(list_of_features: list, columns: list):
    infos = ''
    for id in list_of_features:
        infos = infos + '\n' + columns[id]
    return infos

# draw confusion matrix
def plot_confusion_metrics(y_test, y_pred, display_labels: list=None, file_path=None):
    # system_confusion_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1),
    #                                            labels=labels)
    # cm_display = ConfusionMatrixDisplay(system_confusion_matrix, display_labels=display_labels)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # cm_display.plot(cmap="RdYlGn", ax=ax)
    # if file_path is not None:
    #     plt.savefig(file_path)
    # plt.show()
    cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
    if display_labels is None:
        display_labels = np.unique(y_test)

    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='g', cmap='RdYlGn', xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()



# get fp, tp, fn, tn
def hipotese_tests(true_y, pred_y):
    tp = TruePositives()
    tp.update_state(true_y, pred_y)
    tp = tp.result().numpy()

    tn = TrueNegatives()
    tn.update_state(true_y, pred_y)
    tn = tn.result().numpy()

    fp = FalsePositives()
    fp.update_state(true_y, pred_y)
    fp = fp.result().numpy()

    fn = FalseNegatives()
    fn.update_state(true_y, pred_y)
    fn = fn.result().numpy()

    return tn, fp, fn, tp


def find_eer(far, frr):
    x = np.absolute((np.array(far) - np.array(frr)))

    y = np.nanargmin(x)
    # print("index of min difference=", y)
    far_optimum = far[y]
    frr_optimum = frr[y]
    return [np.nanargmin(x), max(far_optimum, frr_optimum)]

"""


def draw_far_frr(far, frr, eer, thresholds, plot_title=None, file_path=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(thresholds, far, color='pink', label='FAR (False Acceptance Rate)', linewidth=2)
    ax.plot(thresholds, frr, color='steelblue', label='FRR (False Rejection Rate)', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Percentage of tries')

    # Find the threshold where the difference between FAR and FRR is the smallest
    diffs = np.abs(np.array(far) - np.array(frr))
    min_index = np.argmin(diffs)

    plt.scatter(thresholds[min_index], eer, color='red', s=100, zorder=5)
    plt.annotate(f'EER: {eer:.4f}', (thresholds[min_index], eer), textcoords="offset points",
                 xytext=(0, 10), ha='center')

    ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
              bbox_transform=fig.transFigure, ncol=3, fontsize=8)
    if plot_title is not None:
        ax.set_title(plot_title)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()


# get eer
def calculate_far_frr_eer(y_test, y_pred, bins=100):
    thresholds = np.linspace(0, 1, bins)
    far_values = []
    frr_values = []

    for threshold in thresholds:
        # Classify predictions
        positive_preds = y_pred >= threshold
        negative_preds = y_pred < threshold

        # Calculate false accepts and false rejects
        false_accepts = np.sum((y_test == 0) & (positive_preds))
        false_rejects = np.sum((y_test == 1) & (negative_preds))

        # Calculate total impostor and genuine attempts
        total_impostor_attempts = np.sum(y_test == 0)
        total_genuine_attempts = np.sum(y_test == 1)

        # Calculate FAR and FRR
        far = false_accepts / total_impostor_attempts
        frr = false_rejects / total_genuine_attempts

        far_values.append(far)
        frr_values.append(frr)

    diffs = np.abs(np.array(far_values) - np.array(frr_values))
    min_index = np.argmin(diffs)

    # The EER is approximately the average of FAR and FRR at this threshold
    return far_values, frr_values, find_eer(far_values, frr_values), thresholds
"""

# calculate far, frr, eer
def calculate_far_frr_eer(y_test, y_pred, bins=100):
    frr, far = [], []
    threshold = [k / bins for k in range(bins + 1)]
    for thresh in threshold:
        far_counter, frr_counter = 0, 0
        for k in range(y_pred.shape[0]):
            y_prediction, true = y_pred[k], y_test[k]
            if y_prediction.max() > thresh and np.argmax(y_prediction) != np.argmax(true):
                far_counter += 1
            if y_prediction.max() < thresh and np.argmax(y_prediction) == np.argmax(true):
                frr_counter += 1
        far.append(far_counter / y_pred.shape[0])
        frr.append(frr_counter / y_pred.shape[0])
    eer = find_eer(far, frr)
    eer[0] = eer[0] / bins
    return far, frr, eer, threshold





# draw far, frr with eer
def draw_far_frr(far, frr, eer, threshold, plot_title=None, file_path=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(threshold, far, color='pink', label='FAR (False Acceptance Rate)', linewidth=2)
    ax.plot(threshold, frr, color='steelblue', label='FRR (False Rejection Rate)', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Percentage of tries')
    plt.plot(eer[0], eer[1], color='red', marker='o')
    plt.text(eer[0], eer[1], f'    {eer} - EER (Equal Error Rate)', fontdict={'size': 6})
    ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
              bbox_transform=fig.transFigure, ncol=3, fontsize=8)
    if plot_title is not None:
        ax.set_title(plot_title)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()






# draw roc curve
def draw_system_t_roc_curve(far, frr, eer, plot_title=None, file_path=None):
    plt.figure()
    plt.plot(frr, far, color='steelblue', linewidth=2)
    plt.xlabel('FRR (False Rejection Rate)')
    plt.ylabel('FAR (False Acceptance Rate)')
    plt.plot(eer[1], eer[1], color='red', marker='o', label='EER')
    plt.text(eer[1], eer[1], '  EER (Equal Error Rate)', fontdict={'size': 7})
    if plot_title is not None:
        plt.title(plot_title)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()

# draw roc curve
def draw_system_roc_curve(far, frr, eer, plot_title=None, file_path=None):
    plt.figure()
    tpr = 1 - np.array(frr)
    plt.plot( np.sort(np.concatenate([far, [0.0, 1.0]])), np.sort(np.concatenate([tpr, [0.0, 1.0]])), color='steelblue', linewidth=2)

    plt.plot(far, far, color='grey', linestyle='dashed')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(eer[1], 1.0 - eer[1], color='red', marker='o', label='EER')
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.text(eer[1], 1.0 - eer[1], '  EER (Equal Error Rate)', fontdict={'size': 7})
    if plot_title is not None:
        plt.title(plot_title)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()


def save_to_csv(file_path: str, my_dict: dict):
    with open(file_path, 'a+', encoding='utf-8') as file:
        w = DictWriter(file, my_dict.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(my_dict)

