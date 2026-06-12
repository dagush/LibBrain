import os
import matplotlib.pyplot as plt

def plot_stemplot_RSN(subject,
                      proj,
                      RSN_names,
                      y_label,
                      save_path):
    """
    Generates and saves stem plots for each RSN.

    :param subject: subject ID
    :param proj: projection matrix (n_RSNs x num_harmonics)
    :param RSN_names: list of RSN names
    """
    save_folder = os.path.join(save_path, "images", f"{subject}_RSN")
    os.makedirs(save_folder, exist_ok=True)

    for i, rsn_name in enumerate(RSN_names):
        rsn_folder = os.path.join(save_folder, rsn_name, y_label)
        os.makedirs(rsn_folder, exist_ok=True)

        plt.figure(figsize=(6, 4))
        plt.stem(proj[i, 0:40])
        plt.title(f"{rsn_name} - {y_label}")
        plt.xlabel("N harmonic")
        plt.ylabel(y_label)

        image_path = os.path.join(rsn_folder, f"{rsn_name}_{y_label}.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Stem plot saved to", image_path)
