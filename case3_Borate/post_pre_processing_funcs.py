# IMPORTING LIBRARIES
###########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Loading the Plotting Libraries
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image
from scipy import integrate
import plotly.graph_objects as go
###########################################
# IMPORTING MY OWN FUNCTIONS
###########################################
from SMB_func import SMB




# CONCENTRATION PROFILES
###########################################################################################
def see_prod_curves(t_odes, Y, t_index) :
    # Y = C_feed, C_raff, C_ext
    # X = t_sets
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    

    # 0 - Feed Profile
    # 1 - Raffinate Profile
    # 2 - Extract Profile
    
    # Concentration Plots
    for i in range(num_comp): # for each component
        if iso_type == "UNC":
            ax[0].plot(t_odes[i], Y[0][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            ax[1].plot(t_odes[i], Y[1][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            ax[2].plot(t_odes[i], Y[2][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
        
        elif iso_type == "CUP":    
            ax[0].plot(t_odes[i], Y[0][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            ax[1].plot(t_odes[i], Y[1][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            ax[2].plot(t_odes[i], Y[2][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
        
    # Add Accessories
    ax[0].set_xlabel('Time, s')
    ax[0].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[0].set_title(f'Feed Concentration Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    ax[0].legend()

    ax[1].set_xlabel('Time, s')
    ax[1].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[1].set_title(f'Raffinate Elution Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    ax[1].legend()

    ax[2].set_xlabel('Time, s')
    ax[2].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[2].set_title(f'Extract Elution Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    ax[2].legend()


    plt.show()

    # Volumetric Flowrate Plots
    fig, vx = plt.subplots(1, 2, figsize=(25, 5))
    for i in range(num_comp): # for each component
        if iso_type == "UNC":
            
            vx[0].plot(t_odes[i], Y[3][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            vx[1].plot(t_odes[i], Y[4][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
        
        elif iso_type == "CUP":    
            
            vx[0].plot(t_odes[i], Y[3][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
            vx[1].plot(t_odes[i], Y[4][i], color = color[i], label = f"{Names[i]}, H{Names[i]}:{parameter_sets[i]['H']}, kh:{parameter_sets[i]['kh']}")
        
    # Add Accessories
    vx[0].set_xlabel('Time, s')
    vx[0].set_ylabel('($\mathregular{cm^3/s}$)')
    vx[0].set_title(f'Raffinate Volumetric Flowrates')
    vx[0].legend()

    vx[1].set_xlabel('Time, s')
    vx[1].set_ylabel('($\mathregular{cm^3/s}$)')
    vx[1].set_title(f'Extract Volumetric Flowrates')
    vx[1].legend()

    plt.show()

def col_liquid_profile(t, y, Axis_title, c_in, Ncol_num, L_total):
    y_plot = np.copy(y)
    # # Removeing the BC nodes
    # for del_row in start:
    #     y_plot = np.delete(y_plot, del_row, axis=0)
        
    # print('y_plot:', y_plot.shape)
    
    x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
    dt = t[1] - t[0]
    

    
    # Start vs End Snapshot
    fig, ax = plt.subplots(1, 2, figsize=(25, 5))

    ax[0].plot(x, y_plot[:, 0], label="t_start")
    ax[0].plot(x, y_plot[:, -1], label="t_end")

    # Add vertical black lines at positions where i % nx_col == 0
    for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
        x_pos = col_idx #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
        #x_pos = dx * x_pos
        ax[0].axvline(x=x_pos, color='k', linestyle='-')
        ax[1].axvline(x=x_pos, color='k', linestyle='-')

    ax[0].set_xlabel('Column Length, m')
    ax[0].set_ylabel('($\mathregular{g/l}$)')
    ax[0].axhline(y=c_in, color='g', linestyle= '--', linewidth=1, label="Inlet concentration")  # Inlet concentration
    ax[0].legend()

    # Progressive Change at all ts:
    for j in range(np.shape(y_plot)[1]):
        ax[1].plot(x, y_plot[:, j])
        ax[1].set_xlabel('Column Length, m')
        ax[1].set_ylabel('($\mathregular{g/l}$)')
    plt.show()


def col_solid_profile(t, y, Axis_title, Ncol_num, start, L_total):
    
    # Removeing the BC nodes
    y_plot = np.copy(y)
    # Removeing the BC nodes
    for del_row in start:
        y_plot = np.delete(y_plot, del_row, axis=0)
        
    # print('y_plot:', y_plot.shape)
    
    x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
    dt = t[1] - t[0]
    
    # Start vs End Snapshot
    fig, ax = plt.subplots(1, 2, figsize=(25, 5))

    ax[0].plot(x, y_plot[:, 0], label="t_start")
    ax[0].plot(x, y_plot[:, -1], label="t_end")
    # ax[0].plot(x, y_plot[:, len(t) // 2], label="t_middle")

    # Add vertical black lines at positions where i % nx_col == 0
    for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
        x_pos = col_idx*L #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
        #x_pos = dx * x_pos
        ax[0].axvline(x=x_pos, color='k', linestyle='-')
        ax[1].axvline(x=x_pos, color='k', linestyle='-')

    ax[0].set_xlabel('Column Length, m')
    ax[0].set_ylabel('($\mathregular{g/l}$)')
    ax[0].set_title(f'{Axis_title}')
    ax[0].legend()

    # Progressive Change at all ts:
    for j in range(np.shape(y_plot)[1]):
        ax[1].plot(x, y_plot[:, j])
        ax[1].set_xlabel('Column Length, m')
        ax[1].set_ylabel('($\mathregular{g/l}$)')
        ax[1].set_title(f'{Axis_title}')
    plt.show()  # Display all the figures 



# ANIMATION
###########################################################################################

def coupled_animate_profiles(t, title, y, nx, labels, colors, t_start_inject_all, t_index, L_total, parameter_sets, Ncol_num):
    def create_animation(y_profiles, t, concentration_type, filename, labels, colors,L_total,  parameter_sets, Ncol_num):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Initialize lines for each profile
        lines = []
        x = np.linspace(0, L_total, np.shape(y_profiles[0])[0])
        for i, y_profile in enumerate(y_profiles):
            line, = ax.plot(x, y_profile[:, 0], label=f"{labels[i]}: H{labels[i]} = {parameter_sets[i]['H']}, kh{labels[i]} = {parameter_sets[i]['kh']}", color=colors[i])
            lines.append(line)

        # Add a text box in the top right corner to display the time
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Add black vertical lines at the left edge at all times
        for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
            x_pos = col_idx * L
            ax.axvline(x=x_pos, color='k', linestyle='-')

        # Function to add red vertical lines at the injection times
        def add_pulse_lines(t):
            for col in range(len(t_start_inject_all)):
                for start in t_start_inject_all[col]:
                    if start <= t < start + t_index:
                        x_pos = col * L
                        ax.axvline(x=x_pos, color='r', linestyle='-', linewidth=1)

        # Function to update the y data of the lines
        def update(frame):
            for i, y_profile in enumerate(y_profiles):
                lines[i].set_ydata(y_profile[:, frame])
            time_text.set_text(f'Time: {t[frame]:.2f} s')
            
            # Clear existing red lines
            [line.remove() for line in ax.lines if line.get_color() == 'r']
            # Add new red lines
            add_pulse_lines(t[frame])

            return lines + [time_text]

        # Set the limits for the x and y axis
        y_min = np.min([np.min(y_profile) for y_profile in y_profiles])
        y_max = np.max([np.max(y_profile) for y_profile in y_profiles]) + (5 / 10000000)  # c_IN
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Column Length, m")
        ax.set_ylabel(f"{title} {concentration_type} ($\mathregular{{g/l}}$)")
        ax.legend()

        # Determine the number of frames based on the length of the time vector
        n_frame = len(t)
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_frame), interval=100, blit=True)

        # Set up the writer
        ffmpegWriter = animation.writers['ffmpeg']
        writer = ffmpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        print(f"Saving animation to {filename}...")
        ani.save(filename, writer=writer)
        print(f"Animation saved to {filename}.")

        # Display the animation
        plt.show()

    # Separate the y data into liquid and solid concentrations
    liquid_profiles = [y_profile[:nx, :] for y_profile in y]
    solid_profiles = [y_profile[nx:, :] for y_profile in y]

    # Create animations for liquid and solid concentrations
    create_animation(liquid_profiles, t, "Liquid Concentration", f"{title}_liquid.mp4", labels, colors)
    create_animation(solid_profiles, t, "Solid Concentration", f"{title}_solid.mp4", labels, colors)


def animate_profiles(t_sets, title, y, nx, labels, colors, t_start_inject_all, t_index,L_total,parameter_sets,  Ncol_num, L):
    def create_animation(y_profiles, t_profiles, concentration_type, filename, labels, colors):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Initialize lines for each profile
        lines = []
        x = np.linspace(0, L_total, np.shape(y_profiles[0])[0])
        for i, y_profile in enumerate(y_profiles):
            line, = ax.plot(x, y_profile[:, 0], label=f"{labels[i]}: H{labels[i]} = {parameter_sets[i]['H']}, kh{labels[i]} = {parameter_sets[i]['kh']}", color=colors[i])
            lines.append(line)

        # Add a text box in the top right corner to display the time
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Add black vertical lines at the left edge at all times
        for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
            x_pos = col_idx * L
            ax.axvline(x=x_pos, color='k', linestyle='-')

        # Function to add red vertical lines at the injection times
        def add_pulse_lines(t):
            for col in range(len(t_start_inject_all)):
                for start in t_start_inject_all[col]:
                    if start <= t < start + t_index:
                        x_pos = col * L
                        ax.axvline(x=x_pos, color='r', linestyle='-', linewidth=1)

        # Function to update the y data of the lines
        def update(frame):
            for i, (y_profile, t_profile) in enumerate(zip(y_profiles, t_profiles)):
                if frame < len(t_profile):
                    lines[i].set_ydata(y_profile[:, frame])
                    # time_text.set_text(f'Time:{t_profile[frame]:.2f}s\nCycles:{np.round(t_profile[frame]:.2f)}\n {n_1_cycle}s/cycle: ')
                    time_text.set_text(f'Time: {t_profile[frame]:.2f}s\nCycles: {np.round(t_profile[frame]/n_1_cycle, 1)}\nIndex Time:{t_index}s\n{n_1_cycle/60} min/cycle')

                                    
                    # Clear existing red lines
                    [line.remove() for line in ax.lines if line.get_color() == 'r']
                    # Add new red lines
                    add_pulse_lines(t_profile[frame])
            return lines + [time_text]

        # Set the limits for the x and y axis
        y_min = np.min([np.min(y_profile) for y_profile in y_profiles])
        y_max = np.max([np.max(y_profile) for y_profile in y_profiles]) + (5 / 10000000)  # c_IN
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Column Length, m")
        ax.set_ylabel(f"{title} {concentration_type} ($\mathregular{{g/l}}$)")
        ax.legend()

        # Determine the maximum number of frames
        n_frame = max(len(t_profile) for t_profile in t_profiles)
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_frame), interval=100, blit=True)

        # Set up the writer
        ffmpegWriter = animation.writers['ffmpeg']
        writer = ffmpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        print(f"Saving animation to {filename}...")
        ani.save(filename, writer=writer)
        print(f"Animation saved to {filename}.")

        # Display the animation
        plt.show()

    # Separate the y data into liquid and solid concentrations
    liquid_profiles = [y_profile[:nx, :] for y_profile in y]
    solid_profiles = [y_profile[nx:, :] for y_profile in y]

    # Create animations for liquid and solid concentrations
    create_animation(liquid_profiles, t_sets, "Liquid Concentration", f"{title}_liquid.mp4", labels, colors)
    create_animation(solid_profiles, t_sets, "Solid Concentration", f"{title}_solid.mp4", labels, colors)





#######################################################

# INPUTS

#######################################################



# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "UNC" 

###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Sucrose", "Fructose"]#, "C"]#, "D", "E", "F"]
color = ["g", "orange"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.40         # bed voidage
nx_per_col = 5

Bm = 300

# Column Dimensions

# How many columns in each Zone?
Z1, Z2, Z3, Z4 = 2, 3, 2, 1
zone_config = np.array([Z1, Z2, Z3, Z4])
L = 30 # cm # Length of one column
d_col = 2.6 # cm # column diameter
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 0.2 * d_col # cm

# Time Specs

t_index_min = 198/60 # min # Index time # How long the pulse holds before swtiching
n_num_cycles = 12    # Number of Cycles you want the SMB to run for

###############  FLOWRATES   #################################################################################

# Jochen et al:
Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac

Q_I, Q_II, Q_III, Q_IV = Q_R,  Q_S, Q_P, Q_Q
Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])
# Parameter Sets for different components
################################################################

# Units:
# - Concentrations: g/cm^3
# - kh: 1/s
# - Da: cm^2/s


# parameter_sets = [
# {"kh": 3.15/100, "H": 0.27, "C_feed": 1},  # Component A
# {"kh": 2.217/100, "H": 0.53, "C_feed": 1}] #, # Component B
Pe_all = [165.93, 135.25] # [A, B]
# Pe_all = [165.93, 165.93] # [A, B]
parameter_sets = [
{"kh": 0.0195, "H": 0.29, "C_feed": 1},  # Component A
{"kh": 0.004, "H": 0.38, "C_feed": 1}] #, # Component B

# ISOTHERM PARAMETERS
###########################################################################################
theta_lin = [parameter_sets[i]['H'] for i in range(num_comp)] # [HA, HB]
print('theta_lin:', theta_lin)
# theta_lang = [1, 2, 3, 4 ,5, 6] # [HA, HB]
# theta_cup_lang = [5.29, 3.24, 2.02, 0.03] # [HA, HB, KA, KB]
# theta_fre = [1.2, 0.5]
# theta_blang = [[2.69, 0.0336, 0.0466, 0.1, 1, 3],\
#                 [3.73, 0.0336, 0.0466, 0.3, 1, 3]] # [HA, HB]
# theta_blang = [
#     [2.69, 0.0336, 0.0466, 0.1, 1, 3],
#     [3.73, 0.0336, 0.0466, 0.3, 1, 3],
#     [2.5, 0.045, 0.05, 0.2, 1.2, 2.8],
#     [3.0, 0.038, 0.042, 0.25, 1.1, 2.5],
#     [2.8, 0.036, 0.048, 0.15, 1.3, 3.1],
#     [3.2, 0.040, 0.045, 0.22, 1.4, 2.9]
# ]

#######################################################

# FUNCTION EXECUTIONS

#######################################################

# call the SMB func 
# SMB setup (modify these accordingly)
SMB_inputs = [iso_type, Names, color, num_comp, nx_per_col, e, Pe_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets]
SMB_inputs_names = ['iso_type', 'Names', 'color', 'num_comp', 'nx_per_col', 'e', 'Pe', 'Bm', 'zone_config', 'L', 'd_col', 'd_in', 't_index_min', 'n_num_cycles', 'Q_internal', 'parameter_sets']

y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = SMB(SMB_inputs)



print("-----------------------------------------------------------")
Y = [C_feed, raff_cprofile, ext_cprofile, raff_vflow, ext_vflow]

# Y = [C_feed, ext_vflow, raff_vflow ]
if iso_type == "UNC":
    see_prod_curves(t_sets, Y, t_index_min*60)
elif iso_type == "CUP":
    see_prod_curves(t, Y, t_index_min*60)

# Define the data for the table
data = {
    'Metric': [
        'Total Expected Acc (IN-OUT)', 
        'Total Model Acc (r+l)', 
        'Total Error Percent (relative to Exp_Acc)', 
        'Final Raffinate Collected Purity [A, B,. . ]', 
        'Final Extract Collected Purity [A, B,. . ]',
        'Final Raffinate Recovery[A, B,. . ]', 
        'Final Extract Recovery[A, B,. . ]'
    ],
    'Value': [
        f'{sum(Expected_Acc)} g', 
        f'{sum(Model_Acc)} g', 
        f'{Error_percent} %', 

        f'{raff_intgral_purity} %', 
        f'{ext_intgral_purity} %', 
        f'{raff_recov} %', 
        f'{ext_recov} %'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


# Plot the table as a figure
fig, ax = plt.subplots(figsize=(8, 4)) # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Format the table's appearance
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Adjust scaling of the table

# Display the table
plt.show()


# print("-----------------------------------------------------------")
# print("Starting Animation. . . ")
# if iso_type == "UNC":
#     animate_profiles(t_sets, "4_col", y_matrices, nx, Names, color, t_schedule, t_index_min/60)
# elif iso_type == "CUP":
#     coupled_animate_profiles(t, "4_col", y_matrices, nx, Names, color, t_schedule, t_index_min/60)
# print("\nEnd of Simulation. . . . ")