import numpy as np

def calculate_decay_constant(r90_hrs: float):
    """
    Calculate decay constant for use in TM-21 model.

    Parameters:
    - r90_hrs: Time in hours when brightness drops to 90%.
    """

    r"""
    Convert half-life to decay constant k using:
    $$
        L(r90\_hours) = L_0 * e^{-k \cdot r90\_hours} = 0.9L_0
    $$
    $$
        e^{-k \cdot r90\_hours} = 0.9 = e^{\ln(0.9)}
    $$
    $$
        0.9 = e^{-k \cdot r90\_hours} = e^{\ln(0.9)}
    $$
    $$
        -k = \frac{\ln(0.9)}{r90\_hours}
    $$
    """
    decay_k = np.log(1/0.9) / r90_hrs
    return decay_k

def calculate_relative_decay(decay_k: float, start_hours: float, end_hours: float):
    """
    Calculate the relative decay between two time points.
    This is used to calculate the relative brightness at a given time.
    
    Parameters:
    - decay_k: Decay constant.
    - start_hours: Start time in hours.
    - end_hours: End time in hours.
    """
    # np.exp(-decay_k * end_hours) / np.exp(-decay_k * start_hours) but optimised
    return np.exp(decay_k*(start_hours-end_hours))
