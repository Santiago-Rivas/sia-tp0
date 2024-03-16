from src.catching import attempt_catch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from src.pokemon import StatusEffect


def plot_pokeball_means(pokeballs, pokemons, stats_df):
    pokeballs_mean = {}
    pokeballs_std = {}

    for pokeball in pokeballs:
        pokeballs_mean[pokeball] = round(
            stats_df[pokeball].mean(), 10)
        pokeballs_std[pokeball] = round(
            stats_df[pokeball].std(), 10)

    print(pokeballs_mean)

    # Creating the bar plot
    # plt.bar(pokeballs, pokeballs_mean.values(), color='skyblue')
    plt.bar(pokeballs, pokeballs_mean.values(), color='skyblue',
            yerr=pokeballs_std.values(), capsize=5)

    # Adding title and labels
    plt.title('Mean probability for Different Pokeballs')
    plt.xlabel('Pokeballs')
    plt.ylabel('Probability')

    # Displaying the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


def plot_heavy(pokeballs, stats_df, error_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []
    y_errors = []

    # x_values.append(index.weight)
    # y_values.append(value/stats_df.at[index, "pokeball"])
    # hue_values.append(pokeball)
    # y_errors.append(error_df.at[index, pokeball])

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "heavyball"):
                pokeball_value = stats_df.at[index, "pokeball"]
                norm_value = value / pokeball_value
                x_values.append(index.weight)
                y_values.append(norm_value)
                hue_values.append(pokeball)

                # Calculate error using error propagation formula for division
                value_error = error_df.at[index, pokeball]
                pokeball_error = error_df.at[index, "pokeball"]
                y_error = np.abs(norm_value) * np.sqrt(((value_error / value)
                                                        ** 2) + ((pokeball_error / pokeball_value)**2))
                y_errors.append(y_error)

    # Create scatter plot with all series
    sns.scatterplot(x=x_values, y=y_values, hue=hue_values,
                    markers=False, marker='o', s=15, edgecolor='k')

    plt.errorbar(x=x_values, y=y_values, yerr=y_errors,
                 fmt='none', c='grey', zorder=-1, capsize=2)

    # Add title and labels
    plt.title('Effectiveness of the Heavyball compared to the Pokeball')
    plt.xlabel('Weight')
    plt.ylabel('Effectiveness')

    # Show the plot
    plt.legend(title='Pokeballs')
    plt.show()


def plot_fast(pokeballs, stats_df, error_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []
    y_errors = []

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "fastball"):
                pokeball_value = stats_df.at[index, "pokeball"]
                norm_value = value / pokeball_value
                x_values.append(index.stats[-1])
                y_values.append(norm_value)
                hue_values.append(pokeball)

                # Calculate error using error propagation formula for division
                value_error = error_df.at[index, pokeball]
                pokeball_error = error_df.at[index, "pokeball"]
                y_error = np.abs(norm_value) * np.sqrt(((value_error / value)
                                                        ** 2) + ((pokeball_error / pokeball_value)**2))
                y_errors.append(y_error)

    # Create scatter plot with all series
    sns.scatterplot(x=x_values, y=y_values, hue=hue_values,
                    markers=True, marker='o', s=15, edgecolor='k')
    plt.errorbar(x=x_values, y=y_values, yerr=y_errors,
                 fmt='none', c='grey', zorder=-1, capsize=2)

    # Add title and labels
    plt.title('Effectiveness of the Fastball compared to the Pokeball')
    plt.xlabel('Speed')
    plt.ylabel('Effectiveness')

    # Show the plot
    plt.legend(title='Pokeballs')
    plt.show()


def plot_ultra(pokeballs, stats_df, error_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []
    y_errors = []
    i = 1

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "ultraball"):
                pokeball_value = stats_df.at[index, "pokeball"]
                norm_value = value / pokeball_value
                x_values.append(i)
                y_values.append(norm_value)
                hue_values.append(pokeball)

                # Calculate error using error propagation formula for division
                value_error = error_df.at[index, pokeball]
                pokeball_error = error_df.at[index, "pokeball"]
                y_error = np.abs(norm_value) * np.sqrt(((value_error / value)**2) + ((pokeball_error / pokeball_value)**2))

                y_errors.append(y_error)
        i = i + 1

    # Create a DataFrame from the x, y, and hue_values lists
    df = pd.DataFrame({'x': x_values, 'y': y_values, 'group': hue_values})

    sns.lmplot(x='x', y='y', data=df, hue='group',
               line_kws={'linewidth': 2}, scatter_kws={'s': 10})

    plt.errorbar(x=x_values, y=y_values, yerr=y_errors,
                 fmt='none', zorder=-1, capsize=2, c="grey")

    # Add title and labels
    plt.title('Effectiveness of the Ultraball compared to the Pokeball')
    plt.xlabel('Pokemon Id')
    plt.ylabel('Effectiveness')

    plt.show()


colors = {
    "pokeball": 'r',
    "fastball": 'g',
    "ultraball": 'b',
    "heavyball": 'y',
}

def plot_status(factory, pokeballs, pokemon_name):  # Agregar como parametros el pokemon
    tries = 100
    attempts = 1000
    pokemon = pokemon_name
    df = pd.DataFrame(columns=["name", "pokeball",
                      "status", "accuracy", "error"])
    for status in StatusEffect:
        health = 1
        new_pokemon = factory.create(pokemon, 100, status, health)
        for pokeball in pokeballs:
            accuracies = []
            for _ in range(tries):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched / attempts)
            df.loc[len(df)] = [pokemon, pokeball, status.name,
                               np.mean(accuracies), np.std(accuracies)]

    display(df)

    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['status'], df_pokeball['accuracy'],
                 color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['status'], df_pokeball['accuracy'],
                     df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs status for ' + pokemon, fontsize=14)
    plt.xlabel('status', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()



def plot_life(factory, pokeballs, pokemon_name):
    attempts = 10
    pokemon = pokemon_name
    df = pd.DataFrame(columns=["name", "pokeball",
                      "health", "accuracy", "error"])
    for i in range(1, 11):
        health = i/10
        new_pokemon = factory.create(pokemon, 100, StatusEffect.NONE, health)
        for pokeball in pokeballs:
            accuracies = []
            for _ in range(10):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(rate)
            df.loc[len(df)] = [pokemon, pokeball, health,
                               np.mean(accuracies), np.std(accuracies)]


    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['health'], df_pokeball['accuracy'],
                 color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['health'], df_pokeball['accuracy'],
                     df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs health for ' + pokemon_name, fontsize=14)
    plt.xlabel('health', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

def plot_level(factory, pokeballs, pokemon_name):  # Agregar como parametros el pokemon
    tries = 100
    attempts = 1000
    pokemon = pokemon_name
    df = pd.DataFrame(columns=["name", "pokeball",
                      "level", "accuracy", "error"])
    for i in range(1, 11):
        health = 1
        level = i*10
        new_pokemon = factory.create(pokemon, level, StatusEffect.NONE, health)
        for pokeball in pokeballs:
            accuracies = []
            for _ in range(10):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched/attempts)
            df.loc[len(df)] = [pokemon, pokeball, level,
                               np.mean(accuracies), np.std(accuracies)]



    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['level'], df_pokeball['accuracy'],
                 color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['level'], df_pokeball['accuracy'],
                     df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs level for ' + pokemon, fontsize=14)
    plt.xlabel('level', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


