import numpy as np

# --- Triangular membership functions ---
def triangular(x, a, b, c):
    """Generic triangular membership function"""
    return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

# --- Adaptive fuzzy membership functions (auto-scaled inputs 0–100) ---
def reward_low(x):  return triangular(x, 0, 25, 50)
def reward_med(x):  return triangular(x, 40, 60, 80)
def reward_high(x): return triangular(x, 70, 100, 130)

def power_low(x):   return triangular(x, 0, 20, 40)
def power_med(x):   return triangular(x, 30, 60, 90)
def power_high(x):  return triangular(x, 70, 100, 130)

def util_low(x):    return triangular(x, 0, 30, 60)
def util_med(x):    return triangular(x, 50, 70, 90)
def util_high(x):   return triangular(x, 80, 100, 120)

# Priority output levels
priority_values = {
    "reject": 20,
    "moderate": 50,
    "strong": 90
}

# --- Fuzzy inference system ---
def fuzzy_priority(reward, power, util):
    """
    Fuzzy-based priority computation for task-server suitability.
    reward, power, util ∈ [0, 100]
    """
    rL, rM, rH = reward_low(reward), reward_med(reward), reward_high(reward)
    pL, pM, pH = power_low(power), power_med(power), power_high(power)
    uL, uM, uH = util_low(util), util_med(util), util_high(util)

    # Rules
    rules = [
        min(rH, pL) * priority_values["strong"],
        min(rH, pH) * priority_values["moderate"],
        min(rL, pH) * priority_values["reject"],
        uH * priority_values["reject"],
        min(rM, pM) * priority_values["moderate"],
        min(rL, pL) * priority_values["moderate"]
    ]

    weights = [
        min(rH, pL),
        min(rH, pH),
        min(rL, pH),
        uH,
        min(rM, pM),
        min(rL, pL)
    ]

    return 0 if sum(weights) == 0 else sum(rules) / sum(weights)
