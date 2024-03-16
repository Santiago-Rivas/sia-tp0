from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
import pandas as pd
import json
import numpy as np

from src.functions import plot_pokeball_means
from src.functions import plot_heavy, plot_fast, plot_ultra
from src.functions import plot_status, plot_life


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")

    # Open the JSON file
    with open('pokemon.json', 'r') as file:
        # Load JSON data
        data = json.load(file)

    pokemon_names = list(data.keys())

    pokemons = []
    for pokemon in pokemon_names:
        pokemons.append(factory.create(pokemon, 100, StatusEffect.NONE, 1))

    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]

    stats_df = pd.DataFrame(0.0, index=pokemons, columns=pokeballs)
    error_df = pd.DataFrame(0.0, index=pokemons, columns=pokeballs)
    iterations = 100

    for pokeball in pokeballs:
        accuracies = []
        for pokemon in pokemons:
            for _ in range(iterations):
                catched = int(attempt_catch(pokemon, pokeball)[0])
                catched = catched / iterations
                stats_df.at[pokemon,
                            pokeball] += catched
                accuracies.append(catched)
            error_df.at[pokemon,
                        pokeball] = np.std(accuracies)

    print(stats_df)

    # Remove all pokemons that were never catched
    error_df = error_df[~(stats_df == 0).any(axis=1)]
    stats_df = stats_df[~(stats_df == 0).any(axis=1)]

    plot_pokeball_means(pokeballs, pokemons, stats_df)
    plot_heavy(pokeballs, stats_df, error_df)
    plot_fast(pokeballs, stats_df, error_df)
    plot_ultra(pokeballs, stats_df, error_df)
    plot_status(factory, "pikachu")
    plot_life(factory, "dragonair")
