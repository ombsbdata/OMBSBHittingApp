import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import gdown
import tempfile


# Configure Streamlit for wide layout and remove margins for better printing
st.set_page_config(
    page_title="Hitter Report",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for better print layout
st.markdown("""
<style>
    /* Remove default Streamlit padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
    
    /* Print-specific styles */
    @media print {
        .main .block-container {
            padding: 0;
            margin: 0;
            max-width: 100%;
            width: 100%;
        }
        
        /* Hide Streamlit UI elements when printing */
        header, .stApp > header, .stSelectbox, .stButton, .stMarkdown h3 {
            display: none !important;
        }
        
        /* Ensure figures use full width */
        .stPlotlyChart, .matplotlib-container {
            width: 100% !important;
            height: auto !important;
        }
        
        /* Page break control */
        .print-break {
            page-break-before: always;
        }
    }
    
    /* Hide selectboxes and buttons when printing */
    @media print {
        .element-container:has(.stSelectbox),
        .element-container:has(.stButton) {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load the CSV data
file_path = 'Fall_2025_wRV_with_stuff.csv'
data = pd.read_csv(file_path, low_memory=False)
data = data[data['BatterTeam'] == 'OLE_REB']

# Load the Ole Miss logo
logo_path = 'OMBSB_Analytics_logo-removebg-preview.png'
logo_img = mpimg.imread(logo_path)

# Standardize AutoPitchType values to ensure consistency
data['AutoPitchType'] = data['AutoPitchType'].str.strip().str.capitalize()

# Ensure the 'Date' column is standardized to a single format (YYYY-MM-DD) and drop invalid rows
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

# Define color palette for PitchCall
pitch_call_palette = {
    'StrikeCalled': 'orange',
    'BallCalled': 'green',
    'BallinDirt': 'green',
    'Foul': 'tan',
    'InPlay': 'blue',
    'FoulBallNotFieldable': 'tan',
    'StrikeSwinging': 'red',
    'BallIntentional': 'green',  # Changed to green to match other balls
    'FoulBallFieldable': 'tan',
    'HitByPitch': 'lime'
}

# Define condensed legend labels for pitch calls
pitch_call_legend_labels = {
    'StrikeCalled': 'Strike Called',
    'BallCalled': 'Ball',
    'BallinDirt': 'Ball',
    'Foul': 'Foul',
    'InPlay': 'In Play',
    'FoulBallNotFieldable': 'Foul',
    'StrikeSwinging': 'Strike Swinging',
    'BallIntentional': 'Ball',
    'FoulBallFieldable': 'Foul',
    'HitByPitch': 'Hit By Pitch'
}

# Define marker styles for AutoPitchType
pitch_type_markers = {
    'Fastball': 'o',
    'Curveball': 's',
    'Slider': '^',
    'Changeup': 'D'
}

# Streamlit app setup
st.title("OMBSB Hitting App")

# ----- Strike-zone geometry (feet) -----
rulebook_left  = -0.83083
rulebook_right =  0.83083
rulebook_bottom = 1.5
rulebook_top    = 3.3775

# “Shadow” ring (your larger box around the zone)
shadow_left   = -0.99750
shadow_right  =  0.99750
shadow_bottom =  1.377
shadow_top    =  3.5


# Create tabs
tab1, tab2 = st.tabs(["Post-Game Summary", "Hitter Profile"])

with tab1:
    # Create columns for better layout control
    col1, col2, col3 = st.columns([2, 2, 1])

    # Get unique dates for selection
    unique_dates = sorted(data['Date'].unique()) if 'Date' in data.columns else []

    # Automatically select the most recent date upon app launch
    with col1:
        if unique_dates:
            default_date = max(unique_dates)  # Most recent date
            selected_date = st.selectbox("Select a Date", options=unique_dates, index=unique_dates.index(default_date))
        else:
            selected_date = None

    # Filter the data based on the selected date first
    filtered_data = data[data['Date'] == selected_date] if selected_date else data

    # Get unique batters from the **filtered** data
    unique_batters = sorted(filtered_data['Batter'].unique()) if not filtered_data.empty else []

    # Create batter selection dropdown with only available batters
    with col2:
        selected_batter = st.selectbox("Select a Batter", options=unique_batters)

    # Add print mode toggle
    with col3:
        print_mode = st.checkbox("Print Mode", help="Optimizes layout for printing")

    # Filter data further based on selected batter
    if selected_batter:
        filtered_data = filtered_data[filtered_data['Batter'] == selected_batter]

    # Generate plot for selected batter
    if not filtered_data.empty:
        plate_appearance_groups = filtered_data.groupby((filtered_data['PitchofPA'] == 1).cumsum())
        num_pa = len(plate_appearance_groups)

        # Adjust figure size based on print mode
        if print_mode:
            fig_width, fig_height = 20, 11  # Larger for print
            title_fontsize = 24
            subtitle_fontsize = 16
            pa_title_fontsize = 18
        else:
            fig_width, fig_height = 15, 8.5  # Original size for web
            title_fontsize = 18
            subtitle_fontsize = 12
            pa_title_fontsize = 14

        # Create the figure with adjusted size
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Adjust GridSpec for better space utilization (no need for legend space now)
        gs = GridSpec(3, 4, figure=fig, 
                      width_ratios=[1.5, 1.5, 1.5, 1.2], 
                      height_ratios=[1, 1, 1])
        gs.update(wspace=0.25, hspace=0.35)

        # Create small plots in the left three columns using GridSpec
        axes = []
        for i in range(min(num_pa, 9)):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(1, 4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1)
            axes.append(ax)

        table_data = []

        # Strike zone and "Heart" of the zone parameters
        strike_zone_width = 17 / 12
        strike_zone_params = {'x_start': -strike_zone_width / 2, 'y_start': 1.5, 'width': strike_zone_width, 'height': 3.3775 - 1.5}
        heart_zone_params = {
            'x_start': strike_zone_params['x_start'] + strike_zone_params['width'] * 0.25,
            'y_start': strike_zone_params['y_start'] + strike_zone_params['height'] * 0.25,
            'width': strike_zone_params['width'] * 0.5,
            'height': strike_zone_params['height'] * 0.5
        }
        shadow_zone_params = {'x_start': -strike_zone_width / 2 - 0.2, 'y_start': 1.3, 'width': strike_zone_width + 0.4, 'height': 3.6 - 1.3}

        for i, (pa_id, pa_data) in enumerate(plate_appearance_groups, start=1):
            if i > 9:
                break

            ax = axes[i - 1]

            # Pitcher Information
            pitcher_throws = pa_data.iloc[0]['PitcherThrows']
            handedness_label = 'RHP' if pitcher_throws == 'Right' else 'LHP'
            pitcher_name = pa_data.iloc[0]['Pitcher']

            # Add the PA number and handedness label above each plot
            ax.set_title(f'PA {i} vs {handedness_label}', fontsize=pa_title_fontsize, fontweight='bold')

            # Add the opposing Pitcher's name under the PA graph
            pitcher_fontsize = 12 if print_mode else 10
            ax.text(0.5, -0.12, f'P: {pitcher_name}', fontsize=pitcher_fontsize, fontstyle='italic', ha='center', transform=ax.transAxes)

            pa_rows = []

            # Draw strike zone and shadow zone
            ax.add_patch(plt.Rectangle((shadow_zone_params['x_start'], shadow_zone_params['y_start']),
                                       shadow_zone_params['width'], shadow_zone_params['height'],
                                       fill=False, color='gray', linestyle='--', linewidth=2))
            ax.add_patch(plt.Rectangle((strike_zone_params['x_start'], strike_zone_params['y_start']),
                                       strike_zone_params['width'], strike_zone_params['height'],
                                       fill=False, color='black', linewidth=2))
            ax.add_patch(plt.Rectangle((heart_zone_params['x_start'], heart_zone_params['y_start']),
                                       heart_zone_params['width'], heart_zone_params['height'],
                                       fill=False, color='red', linestyle='--', linewidth=2))

            for _, row in pa_data.iterrows():
                scatter_size = 200 if print_mode else 150
                sns.scatterplot(
                    x=[row['PlateLocSide']],
                    y=[row['PlateLocHeight']],
                    hue=[row['PitchCall']],
                    palette=pitch_call_palette,
                    marker=pitch_type_markers.get(row['AutoPitchType'], 'o'),
                    s=scatter_size,
                    legend=False,
                    ax=ax
                )
                
                offset = -0.05 if row['AutoPitchType'] == 'Slider' else 0
                pitch_num_fontsize = 10 if print_mode else 8
                ax.text(row['PlateLocSide'], row['PlateLocHeight'] + offset, f"{int(row['PitchofPA'])}",
                    color='white', fontsize=pitch_num_fontsize, ha='center', va='center', weight='bold')

                pitch_speed = f"{round(row['RelSpeed'], 1)} MPH"
                pitch_type = row['AutoPitchType']

                # Extract values for the last pitch
                if row.name == pa_data.index[-1]:  # Check if it's the last pitch in PA
                    play_result = row['PlayResult']
                    kor_bb_result = row['KorBB']
                    pitch_call = row['PitchCall']
                    
                    # Find the first non-"Undefined" value
                    outcome_x = next(
                        (result for result in [play_result, kor_bb_result, pitch_call] if result != "Undefined"),
                        "Undefined"
                    )
                else:
                    outcome_x = row['PitchCall']  # Empty for non-final pitches in the PA

                # Add row to table data
                pa_rows.append([f"Pitch {int(row['PitchofPA'])}", f"{pitch_speed} {pitch_type}", outcome_x])

            table_data.append([f'PA {i}', '', ''])
            table_data.extend(pa_rows)

        # Create consolidated legend entries (remove duplicates and use condensed labels)
        unique_pitch_calls = {}
        for pitch_call, color in pitch_call_palette.items():
            legend_label = pitch_call_legend_labels.get(pitch_call, pitch_call)
            if legend_label not in unique_pitch_calls:
                unique_pitch_calls[legend_label] = color
        
        # Add horizontal legends at the bottom of the figure
        legend_fontsize = 12 if print_mode else 9
        legend_title_fontsize = 14 if print_mode else 11
        legend_markersize = 8 if print_mode else 6

        # Create pitch call legend (colors)
        handles1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=legend_markersize, linestyle='', label=label)
                    for label, color in unique_pitch_calls.items()]
        
        # Create pitch type legend (shapes)
        handles2 = [plt.Line2D([0], [0], marker=marker, color='black', markersize=legend_markersize, linestyle='', label=label)
                    for label, marker in pitch_type_markers.items()]

        # Position legends horizontally at bottom of figure
        legend1 = fig.legend(handles=handles1, title='Pitch Call (Colors)', 
                            loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                            ncol=len(unique_pitch_calls), fontsize=legend_fontsize, 
                            title_fontsize=legend_title_fontsize, frameon=False)
        
        legend2 = fig.legend(handles=handles2, title='Pitch Type (Shapes)', 
                            loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                            ncol=len(pitch_type_markers), fontsize=legend_fontsize, 
                            title_fontsize=legend_title_fontsize, frameon=False)

        # Adjust the bottom margin to accommodate legends
        fig.subplots_adjust(bottom=0.15)

        # Add the pitch-by-pitch table
        ax_table = fig.add_subplot(gs[:, 3])  # Use the last column for the table
        ax_table.axis('off')

        y_position = 1.0
        x_position = 0.05
        table_fontsize = 12 if print_mode else 7
        table_pa_fontsize = 14 if print_mode else 10
        
        for row in table_data:
            if 'PA' in row[0]:  # Highlight plate appearances
                ax_table.text(x_position, y_position, f'{row[0]}', fontsize=table_pa_fontsize, fontweight='bold', fontstyle='italic')
                ax_table.axhline(y=y_position - 0.01, color='black', linewidth=1)
                y_position -= 0.05
            else:  # Add pitch details
                text_str = f"  {row[0]}  |  {row[1]}  |  {row[2]}"
                ax_table.text(x_position, y_position, text_str, fontsize=table_fontsize)
                y_position -= 0.04

        # Add the main title to the figure
        fig.suptitle(f"{selected_batter} Report for {selected_date}", fontsize=title_fontsize, weight='bold')

        # --- Compute Postgame Stats ---
        whiffs = filtered_data['PitchCall'].eq('StrikeSwinging').sum()
        hard_hits = filtered_data[(filtered_data['PitchCall'] == 'InPlay') & (filtered_data['ExitSpeed'] >= 95)].shape[0]
        barrels = filtered_data[(filtered_data['ExitSpeed'] >= 95) & (filtered_data['Angle'].between(10, 35))].shape[0]

        swing_calls = ['Foul', 'InPlay', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable']
        swings = filtered_data[filtered_data['PitchCall'].isin(swing_calls)]
        chase_swings = swings[
            (swings['PlateLocSide'] < -0.7083) | (swings['PlateLocSide'] > 0.7083) |
            (swings['PlateLocHeight'] < 1.5) | (swings['PlateLocHeight'] > 3.3775)
        ]
        chase_count = chase_swings.shape[0]

        # --- Add stat line under the title ---
        fig.text(
            0.5, 0.93 if print_mode else 0.93, 
            f"Whiffs: {whiffs}    Hard Hit: {hard_hits}    Barrels: {barrels}    Chase: {chase_count}", 
            fontsize=subtitle_fontsize, ha='center'
        )

        # Add the Ole Miss logo in the top right corner
        logo_size = 0.12 if print_mode else 0.10
        logo_ax = fig.add_axes([0.85, 0.92, logo_size, logo_size])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

        # Use full width container for the plot
        if print_mode:
            st.markdown('<div class="print-optimized">', unsafe_allow_html=True)
        
        st.pyplot(fig, use_container_width=True)
        
        if print_mode:
            st.markdown('</div>', unsafe_allow_html=True)
            # Add page break before batted ball graphic
            st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)
    else:
        st.write("No data available for the selected filters.")

    # Initialize session state for rotation
    if "rotate_180" not in st.session_state:
        st.session_state.rotate_180 = False

    # Add buttons for rotation and reset (hide in print mode)
    if not print_mode:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rotate 180°"):
                st.session_state.rotate_180 = not st.session_state.rotate_180
        with col2:
            if st.button("Reset"):
                st.session_state.rotate_180 = False

    # Generate the batted ball graphic
    if not print_mode:
        st.markdown("### Batted Ball Locations")

    # Adjust batted ball graphic size for print mode
    bb_fig_size = (5, 5) if print_mode else (6, 6)  # Reduced from (10, 10) to (7, 7) for print
    bb_title_fontsize = 16 if print_mode else 16  # Slightly reduced title size
    bb_legend_fontsize = 8 if print_mode else 10

    fig, ax = plt.subplots(figsize=bb_fig_size)

    # Draw the outfield fence
    LF_foul_pole = 330
    LC_gap = 365
    CF = 390
    RC_gap = 365
    RF_foul_pole = 330
    angles = np.linspace(-45, 45, 500)
    distances = np.interp(angles, [-45, -30, 0, 30, 45], [LF_foul_pole, LC_gap, CF, RC_gap, RF_foul_pole])
    x_outfield = distances * np.sin(np.radians(angles))
    y_outfield = distances * np.cos(np.radians(angles))

    # Rotate the graphic if session state is rotated
    if st.session_state.rotate_180:
        x_outfield, y_outfield = -x_outfield, -y_outfield

    ax.plot(x_outfield, y_outfield, color="black", linewidth=2)

    # Draw the foul lines
    foul_x_left = [-LF_foul_pole * np.sin(np.radians(45)), 0]
    foul_y_left = [LF_foul_pole * np.cos(np.radians(45)), 0]
    foul_x_right = [RF_foul_pole * np.sin(np.radians(45)), 0]
    foul_y_right = [RF_foul_pole * np.cos(np.radians(45)), 0]

    if st.session_state.rotate_180:
        foul_x_left, foul_y_left = [-x for x in foul_x_left], [-y for y in foul_y_left]
        foul_x_right, foul_y_right = [-x for x in foul_x_right], [-y for y in foul_y_right]

    ax.plot(foul_x_left, foul_y_left, color="black", linestyle="-")
    ax.plot(foul_x_right, foul_y_right, color="black", linestyle="-")

    # Draw the infield
    infield_side = 90
    bases_x = [0, infield_side, 0, -infield_side, 0]
    bases_y = [0, infield_side, 2 * infield_side, infield_side, 0]

    if st.session_state.rotate_180:
        bases_x, bases_y = [-x for x in bases_x], [-y for y in bases_y]

    ax.plot(bases_x, bases_y, color="brown", linewidth=2)

    # Plot batted ball locations with PA numbers and Exit Speed
    play_result_styles = {
        "Single": ("blue", "o"),
        "Double": ("purple", "o"),
        "Triple": ("gold", "o"),
        "HomeRun": ("orange", "o"),
        "Out": ("black", "o"),
    }

    scatter_size = 200 if print_mode else 150
    text_fontsize = 12 if print_mode else 10
    speed_fontsize = 10 if print_mode else 8

    for pa_number, pa_data in plate_appearance_groups:
        if pa_data.empty:
            continue
        last_pitch = pa_data.iloc[-1]
        bearing = np.radians(last_pitch["Bearing"])
        distance = last_pitch["Distance"]
        exit_speed = round(last_pitch["ExitSpeed"], 1) if pd.notnull(last_pitch["ExitSpeed"]) else "NA"
        play_result = last_pitch["PlayResult"]

        # Convert polar to Cartesian coordinates
        x = distance * np.sin(bearing)
        y = distance * np.cos(bearing)

        if st.session_state.rotate_180:
            x, y = -x, -y

        # Get play result style
        color, marker = play_result_styles.get(play_result, ("black", "o"))

        # Plot the hit location
        ax.scatter(x, y, color=color, marker=marker, s=scatter_size, edgecolor="black")

        # Flip PA number **visually** by rotating it in place
        pa_rotation = 180 if st.session_state.rotate_180 else 0
        ax.text(
            x, y, str(pa_number), 
            color="white", fontsize=text_fontsize, fontweight="bold", ha="center", va="center",
            rotation=pa_rotation, transform=ax.transData
        )

        # Flip Exit Speed text by rotating it in place
        ev_y_offset = 15 if not st.session_state.rotate_180 else -15
        ev_rotation = 180 if st.session_state.rotate_180 else 0
        ax.text(
            x, y - ev_y_offset, 
            f"{exit_speed} mph" if exit_speed != "NA" else "NA",
            color="red", fontsize=speed_fontsize, fontweight="bold", ha="center",
            rotation=ev_rotation, transform=ax.transData
        )

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("equal")

    # Flip the title text if rotated
    title_rotation = 180 if st.session_state.rotate_180 else 0
    ax.set_title(f"Batted Ball Locations for {selected_batter} (InPlay)", fontsize=bb_title_fontsize, rotation=title_rotation, va="bottom")

    # Add legend for PlayResults
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Single"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", markersize=10, label="Double"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gold", markersize=10, label="Triple"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="HomeRun"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, label="Out"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=bb_legend_fontsize, frameon=False)

    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.15)

    # Display the plot in Streamlit
    st.pyplot(fig, use_container_width=True)

with tab2:
    import os
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.patches import Circle as PatchCircle

    st.subheader("Hitter Profile – Run Value by Zone")

    # --------------------------
    # Helpers scoped to tab2
    # --------------------------
    def first_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    VELO_COL = first_col(data, ["RelSpeed", "ReleaseSpeed", "Velo", "PitchVelo"])
    IVB_COL  = first_col(data, ["InducedVertBreak", "InducedVert", "IVB", "VertBreak", "iVB"])
    HB_COL   = first_col(data, ["HorzBreak", "HorizontalBreak", "HB"])

    def tag_zone_bucket(df):
        x = df["PlateLocSide"].to_numpy()
        y = df["PlateLocHeight"].to_numpy()
    
        zone_w = (rulebook_right - rulebook_left)
        zone_h = (rulebook_top - rulebook_bottom)
        heart_x0 = rulebook_left  + 0.25 * zone_w
        heart_x1 = rulebook_right - 0.25 * zone_w
        heart_y0 = rulebook_bottom + 0.25 * zone_h
        heart_y1 = rulebook_top    - 0.25 * zone_h
    
        in_shadow = (x >= shadow_left) & (x <= shadow_right) & (y >= shadow_bottom) & (y <= shadow_top)
        in_heart  = (x >= heart_x0) & (x <= heart_x1) & (y >= heart_y0) & (y <= heart_y1)
        in_chase  = (x >= -1.75) & (x <= 1.75) & (y >= 1.0) & (y <= 4.0)
    
        zone = np.full(len(df), "Waste", dtype=object)
        zone[in_chase]  = "Chase"
        zone[in_shadow] = "Shadow"
        zone[in_heart]  = "Heart"
    
        return pd.Series(zone, index=df.index, name="ZoneBucket")

    SWING_CALLS = {"Foul", "InPlay", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable"}

    def add_flags(df):
        out = df.copy()
        out["ZoneBucket"] = tag_zone_bucket(out)
        out["IsSwing"] = out["PitchCall"].isin(SWING_CALLS)
        out["IsTake"]  = ~out["IsSwing"]
        return out

    def rv_by_zone(df):
        zones = ["Heart", "Shadow", "Chase", "Waste"]
        rows = []
        for z in zones:
            sub = df[df["ZoneBucket"] == z]
            swings = sub[sub["IsSwing"]]
            takes  = sub[sub["IsTake"]]
            rows.append({
                "Zone": z,
                "Pitches": len(sub),
                "Swing%": (len(swings) / len(sub) * 100) if len(sub) else 0.0,
                "Take%": (len(takes) / len(sub) * 100) if len(sub) else 0.0,
                "RV_swing": swings["run_value"].sum(),
                "RV_take":  takes["run_value"].sum(),
                "RV_total": sub["run_value"].sum()
            })
        out = pd.DataFrame(rows)
        totals = {
            "sw_total": df.loc[df["IsSwing"], "run_value"].sum(),
            "tk_total": df.loc[df["IsTake"],  "run_value"].sum(),
            "total_rv": df["run_value"].sum(),
            "n": int(len(df)),
            "swing_n": int(df["IsSwing"].sum()),
            "take_n": int((~df["IsSwing"]).sum())
        }
        return out, totals

    def create_statcast_graphic(rv_tbl, totals, batter_name, year, league_tbl=None):
        """Create a Statcast-style graphic focusing on zone diagram and stats"""
        fig = plt.figure(figsize=(22, 8))
        
        # Define zone colors matching Statcast
        zone_colors = {
            'Heart': '#E8B4D4',
            'Shadow': '#F4C9A8', 
            'Chase': '#F9ED97',
            'Waste': '#D3D3D3'
        }
        
        # Create grid with zone labels column
        gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 0.5, 0.8, 1.0, 1.0], wspace=0.25)
        ax_zone = fig.add_subplot(gs[0, 0])
        ax_labels = fig.add_subplot(gs[0, 1])  # Zone labels
        ax_freq = fig.add_subplot(gs[0, 2])
        ax_swing = fig.add_subplot(gs[0, 3])
        ax_rv = fig.add_subplot(gs[0, 4])
        
        # === PANEL 1: Zone Diagram (Clean Design) ===
        ax_zone.set_xlim(-2.3, 2.3)
        ax_zone.set_ylim(0.2, 4.3)
        ax_zone.axis('off')
        ax_zone.set_aspect('equal')
        
        # Calculate dimensions
        sz_width = rulebook_right - rulebook_left
        sz_height = rulebook_top - rulebook_bottom
        
        zone_scale = 1.35
        center_x = 0
        
        # Title at top
        ax_zone.text(center_x, 4.15, 'Strike Zone View', fontsize=13, weight='bold', 
                    ha='center', color='#333')
        
        # Draw from outside in for clean layering
        
        # 1. Waste zone (outermost)
        waste_padding = 0.55
        waste_rect = FancyBboxPatch(
            (center_x - (sz_width * zone_scale)/2 - waste_padding, 
             rulebook_bottom - waste_padding),
            sz_width * zone_scale + 2*waste_padding,
            sz_height * zone_scale + 2*waste_padding,
            boxstyle="round,pad=0.1",
            fc=zone_colors['Waste'], ec='#888', lw=3.5, alpha=0.4
        )
        ax_zone.add_patch(waste_rect)
        
        # 2. Chase zone
        chase_padding = 0.32
        chase_rect = FancyBboxPatch(
            (center_x - (sz_width * zone_scale)/2 - chase_padding, 
             rulebook_bottom - chase_padding),
            sz_width * zone_scale + 2*chase_padding,
            sz_height * zone_scale + 2*chase_padding,
            boxstyle="round,pad=0.08",
            fc=zone_colors['Chase'], ec='black', lw=3.5
        )
        ax_zone.add_patch(chase_rect)
        
        # 3. Shadow zone
        shadow_padding = 0.13
        shadow_rect = Rectangle(
            (center_x - (sz_width * zone_scale)/2 - shadow_padding,
             rulebook_bottom - shadow_padding),
            sz_width * zone_scale + 2*shadow_padding,
            sz_height * zone_scale + 2*shadow_padding,
            fc=zone_colors['Shadow'], ec='black', lw=3.5
        )
        ax_zone.add_patch(shadow_rect)
        
        # 4. Strike zone (dashed outline)
        sz_rect = Rectangle(
            (center_x - (sz_width * zone_scale)/2, rulebook_bottom),
            sz_width * zone_scale, sz_height * zone_scale,
            fc='none', ec='black', lw=4, linestyle='--'
        )
        ax_zone.add_patch(sz_rect)
        
        # 5. Heart zone (innermost)
        heart_width = sz_width * 0.5 * zone_scale
        heart_height = sz_height * 0.5 * zone_scale
        heart_x0 = center_x - heart_width/2
        heart_y0 = rulebook_bottom + (sz_height * zone_scale * 0.25)
        
        heart_rect = Rectangle(
            (heart_x0, heart_y0),
            heart_width, heart_height,
            fc=zone_colors['Heart'], ec='#C41E3A', lw=4
        )
        ax_zone.add_patch(heart_rect)
        
        # Get run values
        rv_dict = rv_tbl.set_index('Zone')['RV_total'].to_dict()
        
        # Add ONLY run values to the zones - no zone labels here
        # Heart
        heart_rv = rv_dict.get('Heart', 0)
        ax_zone.text(center_x, heart_y0 + heart_height/2, 
                    f'{heart_rv:+.0f}',
                    fontsize=22, weight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=2.5))
        
        # Shadow
        shadow_rv = rv_dict.get('Shadow', 0)
        shadow_y = rulebook_top + shadow_padding + 0.05
        ax_zone.text(center_x, shadow_y, f'{shadow_rv:+.0f}',
                    fontsize=20, weight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='black', lw=2.5))
        
        # Chase
        chase_rv = rv_dict.get('Chase', 0)
        chase_y = rulebook_bottom - shadow_padding - chase_padding + 0.08
        ax_zone.text(center_x, chase_y, f'{chase_rv:+.0f}',
                    fontsize=20, weight='bold', ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='black', lw=2.5))
        
        # Waste - positioned to the side
        waste_rv = rv_dict.get('Waste', 0)
        waste_x = center_x - (sz_width * zone_scale)/2 - chase_padding - 0.55
        waste_y = (rulebook_bottom + rulebook_top) / 2
        ax_zone.text(waste_x, waste_y, f'{waste_rv:+.0f}',
                    fontsize=18, weight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', lw=2))
        
        # === PANEL 2: Shared Zone Labels ===
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(0, 4)
        ax_labels.axis('off')
        
        # Title space
        ax_labels.text(0.5, 3.8, 'Zone', fontsize=14, weight='bold', ha='center', color='#333')
        
        # Zone labels aligned with rows
        zones_ordered = ['Heart', 'Shadow', 'Chase', 'Waste']
        label_y_positions = [3.0, 2.05, 1.1, 0.15]
        
        for i, zone in enumerate(zones_ordered):
            # Colored background box
            box_height = 0.65
            box = Rectangle((0.05, label_y_positions[i] - box_height/2), 0.9, box_height,
                           fc=zone_colors[zone], ec='black', lw=2.5, alpha=0.8)
            ax_labels.add_patch(box)
            
            # Zone name
            ax_labels.text(0.5, label_y_positions[i], zone,
                          fontsize=15, weight='bold', ha='center', va='center', color='#222')
        
        # === PANEL 3: Pitch Frequency ===
        ax_freq.set_xlim(0, 1)
        ax_freq.set_ylim(0, 4)
        ax_freq.axis('off')
        
        ax_freq.text(0.5, 3.8, 'Pitch\nFrequency', fontsize=14, weight='bold', 
                    ha='center', va='top', linespacing=1.2, color='#333')
        
        total_pitches = rv_tbl['Pitches'].sum()
        freq_y_positions = [3.0, 2.05, 1.1, 0.15]  # Match label positions
        
        for i, zone in enumerate(zones_ordered):
            zone_data = rv_tbl[rv_tbl['Zone'] == zone].iloc[0]
            freq_pct = (zone_data['Pitches'] / total_pitches * 100) if total_pitches > 0 else 0
            
            # Sized bubble
            size = max(0.22, freq_pct / 100 * 0.5)
            circle = PatchCircle((0.5, freq_y_positions[i]), size, 
                                fc=zone_colors[zone], ec='black', lw=3)
            ax_freq.add_patch(circle)
            
            # Count
            ax_freq.text(0.5, freq_y_positions[i], str(zone_data['Pitches']),
                        fontsize=16, weight='bold', ha='center', va='center', color='black')
            
            # Percentage below
            ax_freq.text(0.5, freq_y_positions[i] - size - 0.2, f'{freq_pct:.0f}%',
                        fontsize=13, ha='center', weight='bold')
        
        # Total pitches note at bottom
        ax_freq.text(0.5, -0.15, f'{total_pitches} pitches',
                    fontsize=10, ha='center', style='italic', color='#666')
        
        # === PANEL 4: Swing/Take Percentages ===
        ax_swing.set_xlim(-5, 105)
        ax_swing.set_ylim(0, 4)
        ax_swing.set_xticks([0, 25, 50, 75, 100])
        ax_swing.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)
        ax_swing.set_yticks([])
        ax_swing.spines['left'].set_visible(False)
        ax_swing.spines['top'].set_visible(False)
        ax_swing.spines['right'].set_visible(False)
        ax_swing.spines['bottom'].set_position(('data', 0))
        ax_swing.set_xlabel('Swing                                        Take', fontsize=13, weight='bold', labelpad=8)
        ax_swing.set_title('Swing / Take', fontsize=15, weight='bold', pad=18, color='#333')
        
        swing_y_positions = [3.0, 2.05, 1.1, 0.15]  # Match label positions
        
        for i, zone in enumerate(zones_ordered):
            zone_data = rv_tbl[rv_tbl['Zone'] == zone].iloc[0]
            y_pos = swing_y_positions[i]
            
            swing_pct = zone_data['Swing%']
            take_pct = zone_data['Take%']
            
            # Swing bar (left side)
            ax_swing.barh(y_pos, swing_pct, height=0.6, left=0,
                         color='#5DADE2', alpha=0.9, edgecolor='black', lw=2, zorder=2)
            
            # Take bar (right side)
            ax_swing.barh(y_pos, take_pct, height=0.6, left=swing_pct,
                         color='#E67E22', alpha=0.9, edgecolor='black', lw=2, zorder=2)
            
            # Add percentages inside bars
            if swing_pct > 7:
                ax_swing.text(swing_pct/2, y_pos, f'{swing_pct:.0f}%',
                            fontsize=13, weight='bold', ha='center', va='center', color='white', zorder=3)
            if take_pct > 7:
                ax_swing.text(swing_pct + take_pct/2, y_pos, f'{take_pct:.0f}%',
                            fontsize=13, weight='bold', ha='center', va='center', color='white', zorder=3)
            
            # League average line (if available)
            if league_tbl is not None:
                league_zone = league_tbl[league_tbl['Zone'] == zone]
                if not league_zone.empty:
                    league_swing = league_zone.iloc[0]['Swing%']
                    ax_swing.plot([league_swing, league_swing], [y_pos - 0.35, y_pos + 0.35],
                                 color='#333', lw=3.5, linestyle='--', alpha=0.8, zorder=4)
        
        # League average legend
        if league_tbl is not None:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='#333', lw=3.5, linestyle='--', label='League Avg')]
            ax_swing.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                          frameon=True, fancybox=True, shadow=True)
        
        # === PANEL 5: Run Value (IMPROVED WITH DYNAMIC SCALING) ===
        rv_y_positions = [3.0, 2.05, 1.1, 0.15]  # Match label positions
        
        # Calculate dynamic x-axis limits based on actual data
        all_rv_values = []
        for zone in zones_ordered:
            zone_data = rv_tbl[rv_tbl['Zone'] == zone].iloc[0]
            all_rv_values.extend([zone_data['RV_swing'], zone_data['RV_take']])
        
        # Add league values if available for better scale
        if league_tbl is not None:
            for zone in zones_ordered:
                league_zone = league_tbl[league_tbl['Zone'] == zone]
                if not league_zone.empty:
                    all_rv_values.extend([league_zone.iloc[0]['RV_swing'], league_zone.iloc[0]['RV_take']])
        
        # Calculate smart bounds
        rv_min = min(all_rv_values) if all_rv_values else -10
        rv_max = max(all_rv_values) if all_rv_values else 10
        
        # Add 25% padding and ensure we include 0
        rv_range = rv_max - rv_min
        padding = rv_range * 0.25 if rv_range > 0 else 5
        x_min = min(rv_min - padding, -5)  # At least -5
        x_max = max(rv_max + padding, 5)   # At least +5
        
        # Round to nice numbers
        x_min = np.floor(x_min / 5) * 5
        x_max = np.ceil(x_max / 5) * 5
        
        ax_rv.set_xlim(x_min, x_max)
        ax_rv.set_ylim(0, 4)
        
        # Add subtle gridlines at intervals
        gridline_interval = 10 if (x_max - x_min) > 40 else 5
        for grid_x in np.arange(x_min, x_max + 1, gridline_interval):
            if grid_x == 0:
                ax_rv.axvline(grid_x, color='black', lw=2.5, alpha=0.8, zorder=1)
            else:
                ax_rv.axvline(grid_x, color='#ddd', lw=1.5, alpha=0.6, zorder=1, linestyle='-')
        
        ax_rv.set_yticks([])
        ax_rv.spines['left'].set_visible(False)
        ax_rv.spines['top'].set_visible(False)
        ax_rv.spines['right'].set_visible(False)
        ax_rv.spines['bottom'].set_position(('data', 0))
        ax_rv.tick_params(axis='x', labelsize=11)
        ax_rv.set_xlabel('Runs', fontsize=13, weight='bold', labelpad=8)
        ax_rv.set_title('Run Value', fontsize=15, weight='bold', pad=18, color='#333')
        
        # Plot bars and values
        for i, zone in enumerate(zones_ordered):
            zone_data = rv_tbl[rv_tbl['Zone'] == zone].iloc[0]
            y_pos = rv_y_positions[i]
            
            swing_rv = zone_data['RV_swing']
            take_rv = zone_data['RV_take']
            
            # Swing RV bar with gradient effect (stronger color)
            swing_color = '#27AE60' if swing_rv > 0 else '#E74C3C'
            ax_rv.barh(y_pos + 0.26, swing_rv, height=0.44,
                      color=swing_color, alpha=0.95, edgecolor='black', lw=2, zorder=2)
            
            # Take RV bar with gradient effect (stronger color)
            take_color = '#27AE60' if take_rv > 0 else '#E74C3C'
            ax_rv.barh(y_pos - 0.26, take_rv, height=0.44,
                      color=take_color, alpha=0.95, edgecolor='black', lw=2, zorder=2)
            
            # Add league reference lines if available
            if league_tbl is not None:
                league_zone = league_tbl[league_tbl['Zone'] == zone]
                if not league_zone.empty:
                    league_swing_rv = league_zone.iloc[0]['RV_swing']
                    league_take_rv = league_zone.iloc[0]['RV_take']
                    
                    # Swing reference line
                    ax_rv.plot([league_swing_rv, league_swing_rv], [y_pos + 0.04, y_pos + 0.48],
                             color='#666', lw=2.5, linestyle=':', alpha=0.7, zorder=3)
                    
                    # Take reference line  
                    ax_rv.plot([league_take_rv, league_take_rv], [y_pos - 0.48, y_pos - 0.04],
                             color='#666', lw=2.5, linestyle=':', alpha=0.7, zorder=3)
            
            # Add values with smart positioning
            # Swing values
            bar_width_pct = abs(swing_rv) / (x_max - x_min) * 100
            if bar_width_pct > 15:  # Bar is wide enough for text inside
                text_color = 'white'
                text_x = swing_rv / 2
            else:  # Bar too narrow, put text outside
                text_color = 'black'
                text_x = swing_rv + (3 if swing_rv >= 0 else -3)
            
            ax_rv.text(text_x, y_pos + 0.26, f'{swing_rv:+.0f}',
                      fontsize=14, weight='bold', va='center',
                      ha='center', color=text_color, zorder=4)
            
            # Take values
            bar_width_pct = abs(take_rv) / (x_max - x_min) * 100
            if bar_width_pct > 15:  # Bar is wide enough for text inside
                text_color = 'white'
                text_x = take_rv / 2
            else:  # Bar too narrow, put text outside
                text_color = 'black'
                text_x = take_rv + (3 if take_rv >= 0 else -3)
            
            ax_rv.text(text_x, y_pos - 0.26, f'{take_rv:+.0f}',
                      fontsize=14, weight='bold', va='center',
                      ha='center', color=text_color, zorder=4)
        
        # Enhanced legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#5DADE2', edgecolor='black', label='Swing', alpha=0.9),
            Patch(facecolor='#E67E22', edgecolor='black', label='Take', alpha=0.9)
        ]
        if league_tbl is not None:
            legend_elements.append(Line2D([0], [0], color='#666', lw=2.5, linestyle=':', label='League Avg'))
        
        ax_rv.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                    frameon=True, fancybox=True, shadow=True, ncol=2 if league_tbl is None else 3)
        
        # Add swing/take runs totals at bottom
        swing_total_text = f'{totals["sw_total"]:+.0f} Swing Runs'
        take_total_text = f'{totals["tk_total"]:+.0f} Take Runs'
        ax_rv.text(0.5, -0.15, f'{swing_total_text}   |   {take_total_text}',
                  transform=ax_rv.transAxes, fontsize=13, weight='bold', ha='center',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', 
                           edgecolor='#333', linewidth=2))
        
        # === Overall title ===
        handedness = "RHH" if "RHH" in batter_name else "LHH" if "LHH" in batter_name else ""
        title_text = f'{batter_name} ({handedness}) {year}\n{totals["total_rv"]:+.0f} Run Value'
        fig.suptitle(title_text, fontsize=20, weight='bold', y=0.97)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        return fig

    # --------------------------
    # Controls for this tab
    # --------------------------
    left, mid, right = st.columns([1.2, 1, 1])

    with left:
        batters = sorted(data["Batter"].dropna().unique().tolist())
        sel_batter = st.selectbox("Batter", batters, key='tab2_batter')

        df_b = data[data["Batter"] == sel_batter].copy()
        if "Date" in df_b.columns:
            df_b["Date"] = pd.to_datetime(df_b["Date"], errors="coerce")
            df_b = df_b.dropna(subset=["Date"])
            min_d, max_d = df_b["Date"].min(), df_b["Date"].max()
            date_rng = st.date_input("Date Range", value=[min_d, max_d], 
                                    min_value=min_d, max_value=max_d, key='tab2_date')
            df_b = df_b[(df_b["Date"] >= pd.Timestamp(date_rng[0])) & 
                       (df_b["Date"] <= pd.Timestamp(date_rng[1]))]
            year_display = date_rng[0].year

    with mid:
        ptypes = ["All"] + sorted(df_b["TaggedPitchType"].dropna().unique().tolist())
        sel_ptype = st.selectbox("TaggedPitchType", ptypes, index=0, key='tab2_ptype')
        if sel_ptype != "All":
            df_b = df_b[df_b["TaggedPitchType"] == sel_ptype]

        if VELO_COL:
            vmin, vmax = float(df_b[VELO_COL].min()), float(df_b[VELO_COL].max())
            vel_rng = st.slider("Velocity (mph)", min_value=float(np.floor(vmin)), 
                               max_value=float(np.ceil(vmax)),
                               value=(float(np.floor(vmin)), float(np.ceil(vmax))),
                               key='tab2_velo')
            df_b = df_b[(df_b[VELO_COL] >= vel_rng[0]) & (df_b[VELO_COL] <= vel_rng[1])]

    with right:
        if IVB_COL:
            ivb_min, ivb_max = float(df_b[IVB_COL].min()), float(df_b[IVB_COL].max())
            ivb_rng = st.slider("Induced VB", min_value=float(np.floor(ivb_min)), 
                               max_value=float(np.ceil(ivb_max)),
                               value=(float(np.floor(ivb_min)), float(np.ceil(ivb_max))),
                               key='tab2_ivb')
            df_b = df_b[(df_b[IVB_COL] >= ivb_rng[0]) & (df_b[IVB_COL] <= ivb_rng[1])]

        if HB_COL:
            hb_min, hb_max = float(df_b[HB_COL].min()), float(df_b[HB_COL].max())
            hb_rng = st.slider("Horizontal Break", min_value=float(np.floor(hb_min)), 
                              max_value=float(np.ceil(hb_max)),
                              value=(float(np.floor(hb_min)), float(np.ceil(hb_max))),
                              key='tab2_hb')
            df_b = df_b[(df_b[HB_COL] >= hb_rng[0]) & (df_b[HB_COL] <= hb_rng[1])]

    # --------------------------
    # Build graphic
    # --------------------------
    req_cols = {"PlateLocSide","PlateLocHeight","PitchCall","run_value"}
    if not req_cols.issubset(df_b.columns):
        st.info("Required columns: " + ", ".join(sorted(req_cols - set(df_b.columns))))
    elif df_b.empty:
        st.info("No pitches match the current filters.")
    else:
        df_flags = add_flags(df_b)
        rv_tbl, totals = rv_by_zone(df_flags)

        # Load league data
        league_tbl = None
        SEC_MASTER_GDRIVE_ID = "104xeuMHhMkb18KiJOVu-V48AhMpnmyWT"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                gdown.download(f"https://drive.google.com/uc?id={SEC_MASTER_GDRIVE_ID}", 
                             tmp.name, quiet=True)
                sec = pd.read_csv(tmp.name, low_memory=False)
                
                if sel_ptype != "All" and "TaggedPitchType" in sec.columns:
                    sec = sec[sec["TaggedPitchType"] == sel_ptype]
                if VELO_COL and VELO_COL in sec.columns:
                    sec = sec[(sec[VELO_COL] >= vel_rng[0]) & (sec[VELO_COL] <= vel_rng[1])]
                
                if req_cols.issubset(sec.columns) and not sec.empty:
                    sec_flags = add_flags(sec)
                    league_tbl, _ = rv_by_zone(sec_flags)
        except:
            pass

        fig = create_statcast_graphic(rv_tbl, totals, sel_batter, 
                                     year_display, league_tbl)
        st.pyplot(fig, use_container_width=True)
