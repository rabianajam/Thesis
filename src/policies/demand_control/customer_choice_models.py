import numpy as np
from typing import List, Union, Dict


def simple_customer_choice(restaurant_dict: Dict) -> Union[List[str], None]:
    r"""
    Simple choice model to choose from a dict of restaurants independent of their attributes.
    In this choice model, the customer orders only from single restaurant.
    """
    if not restaurant_dict:
        return None
    # in this simplified choice model, we ignore restaurant attributes and set all restaurant utilities to 0
    restaurant_utilities = np.zeros(len(restaurant_dict.keys()))
    # sample a restaurant
    probability_weights = (np.exp(restaurant_utilities) / (np.sum(np.exp(restaurant_utilities)) + 1)).tolist()
    outside_option_probability = 1 / (np.sum(np.exp(restaurant_utilities)) + 1)
    probability_weights = probability_weights + [outside_option_probability]
    chosen_restaurant = np.random.choice(list(restaurant_dict.keys()) + ["no_choice"], p=probability_weights)
    if chosen_restaurant == "no_choice":
        return None
    return [chosen_restaurant]
