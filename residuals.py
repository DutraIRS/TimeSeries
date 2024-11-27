import pandas as pd
import matplotlib.pyplot as plt

def plot_residual_by_time(data, archive_name):
    num_columns = 4
    num_rows = (len(data.columns) + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 5))

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, column in enumerate(data.columns):
        row = i // num_columns
        col = i % num_columns
        
        axes[row, col].plot(data[column], color='purple')
        axes[row, col].set_title(f"{column}")
        axes[row, col].set_xlabel("Tempo")
        axes[row, col].set_ylabel("Resíduo")

    plt.savefig("results/img/residuos_tempo_" + archive_name + ".png", dpi=300, bbox_inches='tight')

def plot_residual_distribution(data, archive_name):
    num_columns = 4
    num_rows = (len(data.columns) + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 5))

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, column in enumerate(data.columns):
        row = i // num_columns
        col = i % num_columns
        
        axes[row, col].hist(data[column], color='purple')
        axes[row, col].set_title(f"{column}")
        axes[row, col].set_xlabel("Resíduo")
        axes[row, col].set_ylabel("Frequência")

    plt.savefig("results/img/residuos_distribuicao_" + archive_name + ".png", dpi=300, bbox_inches='tight')

def gen_plots(archive_name):
    data = pd.read_json("results/" + archive_name + ".json")
    plot_residual_distribution(data, archive_name)
    plot_residual_by_time(data, archive_name)

def main():
    gen_plots("residual_sem_transformacao")
    gen_plots("residual_normalizacao")
    gen_plots("residual_power_transform")

if __name__ == "__main__":
    main()