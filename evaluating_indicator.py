import csv
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

smooth = 100

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Define the Dice coefficient
def dice_equation(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    if union != 0:
        dices = float((2 * intersection) / union)
    else:
        dices = 0
    return dices


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   Calculate the dice coefficient
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# Set the Width and Height
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount calculate classified results from 0 to n**2-1, and count their frequency of occurrence, then return a matrix with shape (n, n)
    #   These values on diagonal line are correct classified pixel points.
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, png_name_list_2, num_classes=2, name_classes=["_background_", "Tactile_paving"]):
    print('Num classes = ', num_classes)
    # -----------------------------------------#
    #   Ceate a 0-metrix as the orginal confusion matrix
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   Catch the path list of the testing data
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x) for x in png_name_list]
    pred_imgs = [join(pred_dir, x) for x in png_name_list_2]

    dice = 0
    # ------------------------------------------------#
    #   Read each pair of segmented result - ground truth
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   Read each segmented result and translated to be numpy array
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind])) / 255
        # ------------------------------------------------#
        #   Read each ground truth and translated to be numpy array
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind])) / 255

        # If the shapes of these pairs are not consist, then omit them
        if len(label.flatten()) != len(pred.flatten()):
            print(
               'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   Estimate the hist matrix, and accumulate them
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        #   Print the averaged mIoU per 10 images
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.4f}%; mPA-{:0.4f}%; Accuracy-{:0.4f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )

    # ------------------------------------------------#
    #   Calculate the mIoU per type on the testing images
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   Calculate mIoU for each type
    # ------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   Estimate the averaged mIoU got all these types on the testing images, and omit NaN in the calculation
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 4)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 4)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 4)))
    return np.array(hist, int), IoUs, PA_Recall, Precision, dice,


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.4f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.4f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.4f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.4f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.4f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

def main():
    # The path of ground truth results (saved in images)
    gt_dir = args.gt_dir
    # The path of predict results (saved in images)
    pred_dir = args.pred_dir
    txt_dir = args.txt_dir
    count = 0
    png_name_list = []
    with open(os.path.join(txt_dir), 'r') as f:
        file_name = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    for file in file_name:
        count = count + 1
        png_name_list.append(file + ".png")
        # The path of logs
    log_path = args.log_path

    # Log file is named based on the running time
    log_file_name = log_path + 'log-' + 'GRFBUNet' + '.log'
     # save print message
    sys.stdout = Logger(log_file_name, stream=sys.stdout)
    compute_mIoU(gt_dir, pred_dir, png_name_list, png_name_list, num_classes=2, name_classes=["_background_", "Tactile_paving"])

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch GRGB-UNet evaluating")

    parser.add_argument("--gt_dir", default="./data/TP-Dataset/GroundTruth", help="The root of TP-Dataset ground truth list file")
    parser.add_argument("--txt_dir", default="./data/TP-Dataset/Index/predict.txt", help="The root of predicted file")
    parser.add_argument("--pred_dir", default="./predict", help="The root of predicted results of testing samples")
    parser.add_argument("--log_path", default="./log", help="log root")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    main()
