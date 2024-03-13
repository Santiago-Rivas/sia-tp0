from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json


def plot_pokeball_means(pokeballs, pokemons, stats_df):
    pokeballs_mean = {}

    for pokeball in pokeballs:
        pokeballs_mean[pokeball] = round(
            stats_df[pokeball].mean(), 10)

    print(pokeballs_mean)

    # Creating the bar plot
    plt.bar(pokeballs, pokeballs_mean.values(), color='skyblue')

    # Adding title and labels
    plt.title('Mean Stats for Different Pokeballs')
    plt.xlabel('Pokeballs')
    plt.ylabel('Mean Stats')

    # Displaying the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


def plot_heavy(pokeballs, pokemons, stats_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "heavyball"):
                x_values.append(index.weight)
                y_values.append(value/stats_df.at[index, "pokeball"])
                hue_values.append(pokeball)

    # Create scatter plot with all series
    sns.lineplot(x=x_values, y=y_values, hue=hue_values,
                 markers=True, marker='o', markersize=8, ci=None)

    # Add title and labels
    plt.title('Scatter Plot for All Series')
    plt.xlabel('Weight')
    plt.ylabel('Value')

    # Show the plot
    plt.legend(title='Pokeballs')
    plt.show()


def plot_fast(pokeballs, pokemons, stats_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "fastball"):
                x_values.append(index.stats[-1])
                y_values.append(value/stats_df.at[index, "pokeball"])
                hue_values.append(pokeball)

    # Create scatter plot with all series
    sns.lineplot(x=x_values, y=y_values, hue=hue_values,
                 markers=True, marker='o', markersize=8, ci=None)

    # Add title and labels
    plt.title('Scatter Plot for All Series')
    plt.xlabel('Speed')
    plt.ylabel('Value')

    # Show the plot
    plt.legend(title='Pokeballs')
    plt.show()


def plot_ultra(pokeballs, pokemons, stats_df):
    # Create a list to store all x and y values
    x_values = []
    y_values = []
    hue_values = []
    i = 1

    # Iterate over rows of stats_df and store values
    for index, row in stats_df.iterrows():
        for pokeball, value in row.items():
            if (pokeball == "pokeball" or pokeball == "ultraball"):
                x_values.append(i)
                y_values.append(value/stats_df.at[index, "pokeball"])
                hue_values.append(pokeball)
                i = i + 1

    # Create a DataFrame from the x, y, and hue_values lists
    df = pd.DataFrame({'x': x_values, 'y': y_values, 'group': hue_values})

    sns.lmplot(x='x', y='y', data=df, hue='group', ci=None,
               line_kws={'linewidth': 2}, scatter_kws={'s': 20})

    # Add title and labels
    plt.title('Scatter Plot for All Series')
    plt.xlabel('Pokemon Id')
    plt.ylabel('Value')

    # Show the plot
    plt.legend(title='Pokeballs')
    plt.show()


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")

    # Open the JSON file
    with open('pokemon.json', 'r') as file:
        # Load JSON data
        data = json.load(file)

    pokemon_names = list(data.keys())

    pokemons = []
    for pokemon in pokemon_names:
        print(pokemon)
        pokemons.append(factory.create(pokemon, 100, StatusEffect.NONE, 1))

    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
    stats_df = pd.DataFrame(0.0, index=pokemons, columns=pokeballs)
    iterations = 100

    for _ in range(iterations):
        for pokemon in pokemons:
            for pokeball in pokeballs:
                catched = int(attempt_catch(pokemon, pokeball)[0])
                catched = catched / iterations
                stats_df.at[pokemon,
                            pokeball] += catched

    print(stats_df)

    # Remove all pokemons that were never catched
    stats_df = stats_df[~(stats_df == 0).any(axis=1)]

    plot_pokeball_means(pokeballs, pokemons, stats_df)
    plot_heavy(pokeballs, pokemons, stats_df)
    plot_fast(pokeballs, pokemons, stats_df)
    plot_ultra(pokeballs, pokemons, stats_df)
