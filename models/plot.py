import matplotlib.pyplot as plt
from utils import load_pickle_data


def plot_train_loss_acc(model_hist_path):
    # Load history
    train_hist = load_pickle_data(model_hist_path)

    # Prepare paths
    accuracy_plot_path = model_hist_path.split(".pkl")[0] + "acc.png"
    loss_plot_path = model_hist_path.split(".pkl")[0] + "loss.png"

    # Plot and save accuracy
    plt.plot(train_hist["accuracy"])
    plt.title("Évolution de l'exactitude")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train accuracy'], loc='upper left')
    plt.savefig(accuracy_plot_path)
    #plt.show()

    # Plot and save loss
    plt.plot(train_hist["loss"], "r")
    plt.title("Évolution de la fonction de perte")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['train loss'], loc='upper left')
    plt.savefig(loss_plot_path)
    #plt.show()
