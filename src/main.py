from src.state import MealDeliveryMDP
from src.policies.fleet_control.simple_assignment import SimpleAssignmentPolicy
from src.policies.demand_control.simple_proximity import SimpleProximityDemandControl
from src.policies.demand_control.customer_choice_models import simple_customer_choice
import configparser

if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('../data/instances/iowa_110_5_55_80.ini')
    env = MealDeliveryMDP(config, seed=42)
    policy = SimpleAssignmentPolicy()
    demand_policy = SimpleProximityDemandControl(proximity=10*60, restaurant_nodes=env.restaurant_location_list,
                                                 tt_matrix=env.tt_matrix)

    for i in range(0, 100):

        obs = env.reset()

        while True:

            # demand control action
            if env.endogenous_choice and obs["new_customer_info"] is not None:
                demand_action = demand_policy.act(obs)
                obs = env.update_customers_choices(demand_action=demand_action, choice_model=simple_customer_choice)

            # fleet control action and state transition
            action = policy.act(obs)
            obs, cost, done, info = env.step(action)
            if done:
                summary = [i, env.mean_delay, env.total_revenue, env.total_travel_time]
                print("Episode {}. Mean delay {}. Total Revenue {}. Total Travel Time {}.".format(*summary))
                break
