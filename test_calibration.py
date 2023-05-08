import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calib_tools import rmsce

def compute_calibration_curve(conf, correct):
    diff_bins = np.arange(0, 1.1, 0.1)
    pm_accuracy = np.array([]) 
    for i in range(len(diff_bins) - 1):
        lower = diff_bins[i]
        upper = diff_bins[i + 1]
        selected_indices = np.where((conf >= lower) & (conf < upper))[0]
        selected_human_prefs = correct[selected_indices]

        if len(selected_indices) > 0:
            correct_preds = np.sum(selected_human_prefs)
            accuracy = correct_preds / len(selected_indices)
        else:
            accuracy = None

        pm_accuracy = np.append(pm_accuracy, accuracy)
    return diff_bins[:-1] + 0.05, pm_accuracy

k = 5
base_dir = "results"
base_models = [f.split("results_")[1] for f in os.listdir(base_dir) if "results" in f]
models = [os.listdir(f"{base_dir}/results_{base_model}") for base_model in base_models]
full_model_names = [f"{base_model}/{sub_model}" for base_model, sub_models in zip(base_models, models) for sub_model in sub_models]
print(full_model_names)

for model in full_model_names:
    dir = os.path.join(base_dir, "results_{}".format(model))
    fnames = [f for f in os.listdir(dir) if ".csv" in f]

    all_max_probs = []
    all_cors = []
    all_accs = []
    all_confs = []

    for fname in fnames:
        subject = fname.split(".csv")[0]
        fpath = os.path.join(dir, fname)
        df = pd.read_csv(fpath)

        max_probs = []
        cors = []
        for i in range(df.shape[0]):
            probs = [df["{}_choice{}_probs".format(model, choice)][i] for choice in ["A", "B", "C", "D"]]
            cors.append(int(df["{}_correct".format(model)][i]))
            max_probs.append(np.max(probs))
        all_max_probs += max_probs
        all_cors += cors
        all_accs.append(np.mean(cors))
        all_confs.append(np.mean(max_probs))

    avg_max_prob = np.mean(all_max_probs)
    acc = np.mean(all_cors)
    rms_ce = rmsce(np.array(all_cors), np.array(all_max_probs))
    print("{} overall conf: {:.3f}, acc: {:.3f}, RMS: {:.3f}".format(model, avg_max_prob, acc, rms_ce))

    # plt.hist
    # plt.scatter(all_confs, all_accs)
    # min = np.minimum(np.min(all_confs), np.min(all_accs))
    # max = np.maximum(np.max(all_confs), np.max(all_accs))
    # x = np.arange(min, max, 0.01)
    # y = np.arange(min, max, 0.01)
    # plt.plot(x, y, c="r")
    # plt.xlabel("Confidence")
    # plt.ylabel("Accuracy")
    # # plt.savefig("{}/calibration.png".format(dir))
    # plt.savefig(f"{dir}/{model.split('/')[1]}_calibration.png")

    bins, accuracies = compute_calibration_curve(
        np.array(all_confs), 
        np.array(all_cors)
    )
    print(bins)
    print(accuracies)

    plt.scatter(bins, accuracies)

    x_func = np.linspace(0, 1, 1000)
    y_func = x_func
    plt.plot(x_func, y_func, color='black')

    plt.xlabel('P(answer)')
    plt.ylabel('P(correct)')
    plt.title('Calibration Curve')
    plt.savefig(f"{dir}/{model.split('/')[1]}_calibration.png")
    # plt.show()

