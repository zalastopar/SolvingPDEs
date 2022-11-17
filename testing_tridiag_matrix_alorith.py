import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen

# get data
parser = Sbopen()
match_files = [19789, 19794, 19805]
df = pd.concat([parser.event(file)[0] for file in match_files])  # 0 index is the event file

# filter chelsea pressure and pass events
mask_chelsea_pressure = (df.team_name == 'Chelsea FCW') & (df.type_name == 'Pressure')
df_pressure = df.loc[mask_chelsea_pressure, ['x', 'y']]
mask_chelsea_pressure = (df.team_name == 'Chelsea FCW') & (df.type_name == 'Pass')
df_pass = df.loc[mask_chelsea_pressure, ['x', 'y', 'end_x', 'end_y']]