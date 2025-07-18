import json
import matplotlib.pyplot as plt
import numpy as np

def plot_parametric_study_from_json(file_path):
    """
    Load a parametric study JSON file and plot Error % and Simulation time vs varied parameter.
    Assumes JSON was saved using save_output_to_json().
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    param_name = data.get("param_name", "parameter")
    param_values = np.array(data.get("variable", []))
    results = data.get("results", [])

    # Extract metrics
    errors = [r["Error_percent"] for r in results]
    sim_times = [r["Simulation_time"] for r in results]

    # # Infer range if needed
    # if len(param_values) >= 2:
    #     lower_bound = round(min(param_values), 3)
    #     upper_bound = round(max(param_values), 3)
    #     dist_bn_points = round(param_values[1] - param_values[0], 3)
    # else:
    #     lower_bound = upper_bound = param_values[0]
    #     dist_bn_points = None

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(param_values, errors, 'o-', color='tab:red')
    ax[0].set_title(f'Error vs {param_name}')
    ax[0].set_xlabel(param_name)
    ax[0].set_ylabel('Error (%)')

    ax[1].plot(param_values, sim_times, 'o-', color='tab:blue')
    ax[1].set_title(f'Simulation Time vs {param_name}')
    ax[1].set_xlabel(param_name)
    ax[1].set_ylabel('Time (s)')

    # fig.suptitle(f"Parametric Study: {param_name}\nRange: [{lower_bound} to {upper_bound}], Δ = {dist_bn_points}")
    plt.tight_layout()
    plt.show()


plot_parametric_study_from_json("study_L_variation.json")
