
import numpy as np
def apply_scenario(df, scenario):
    if scenario == "Crash":
        df['close'] *= 0.6
    elif scenario == "Bull":
        df['close'] *= 1.2
    elif scenario == "Volatile":
        df['close'] *= (1 + np.random.normal(0,0.04,len(df)))
    return df
