import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Uptime and Cost Optimizer", layout="centered")

if 'n' not in st.session_state: st.session_state.n = 6
if 'k' not in st.session_state: st.session_state.k = 2
if 'r' not in st.session_state: st.session_state.r = 2
if 'l' not in st.session_state: st.session_state.l = 0.5
if 'mu' not in st.session_state: st.session_state.mu = 0.8
if 'stand_by' not in st.session_state: st.session_state.stand_by = 'Warm'
if 'cost_c' not in st.session_state: st.session_state.cost_c = 3.0
if 'cost_r' not in st.session_state: st.session_state.cost_r = 10.0
if 'cost_d' not in st.session_state: st.session_state.cost_d = 100.0

def calculate_uptime(n, k, r, l, mu, stand_by):
    states = n + 1
    P = np.zeros((states, states))

    if stand_by == 'Cold':
        for i in range(states):
            if i > k-1:
                P[i, i - 1] = k * l

    elif stand_by == 'Warm':
        for i in range(states):
            if i > 0:
                P[i, i - 1] = i * l

    for i in range(states):
        if i < states - 1:
            P[i, i + 1] = min(n - i, r) * mu

    for i in range(states):
        P[i, i] = -np.sum(P[i, :])

    eigvals, eigvecs = np.linalg.eig(P.T)
    zero_eig_index = np.argmin(np.abs(eigvals))
    pi = np.real(eigvecs[:, zero_eig_index])
    pi = pi / np.sum(pi)
    P_up = np.sum(pi[k:])

    return P_up

def cost_calculation(n_local, r_local):
    P_up = calculate_uptime(n_local, st.session_state.k, r_local, st.session_state.l, st.session_state.mu,
                            st.session_state.stand_by)
    return st.session_state.cost_c * n_local + st.session_state.cost_r * r_local + st.session_state.cost_d * (1 - P_up)

def visualize_cost_landscape(k, l, mu, stand_by, cost_c, cost_r, cost_d):
    max_n = 40  
    cost_matrix = np.full((max_n - k, max_n), np.nan)

    min_cost = float('inf')
    min_pos = (0, 0)

    for n in range(k, max_n):
        for r in range(1, n):
            cost = cost_calculation(n, r)
            cost_matrix[n - k, r] = cost
            if cost < min_cost:
                min_cost = cost
                min_pos = (n, r)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cost_matrix, cmap="viridis", cbar_kws={'label': 'Total Cost'}, ax=ax, annot=False, fmt=".1f",
                linewidths=0.5, linecolor='gray', mask=np.isnan(cost_matrix))

    # Draw a red box around the minimum
    ax.add_patch(plt.Rectangle((min_pos[1], min_pos[0]), 1, 1, fill=False, edgecolor='red', lw=3))

    ax.set_xlabel("Repairmen (r)")
    ax.set_ylabel("Components (n)")
    ax.set_title("Cost Landscape: Total Cost vs n and r")
    st.pyplot(fig)

def estimate_n_r(k):
    tol = 1e-2
    minimal_cost = float('inf')
    best_combination = (0, 0)

    for n in range(k, 100):
        current_min_cost = float('inf')
        for r in range(1, n):
            cost = cost_calculation(n, r)
            if cost < current_min_cost:
                current_min_cost = cost
            if cost < minimal_cost:
                minimal_cost = cost
                best_combination = (n, r)

        #Early stopping if cost doesn't improve by more than tol
        if abs(current_min_cost - minimal_cost) > tol:
            break

    return best_combination, minimal_cost

st.header("Uptime Calculator")
st.session_state.n = st.number_input("Number of components (n)", min_value=1, value=st.session_state.n)
st.session_state.k = st.number_input("Minimum required components (k)", min_value=1, value=st.session_state.k)
st.session_state.r = st.number_input("Number of repairmen (r)", min_value=1, value=st.session_state.r)
st.session_state.l = st.number_input("Failure rate (lambda)", min_value=0.0, format="%0.2f", value=st.session_state.l)
st.session_state.mu = st.number_input("Repair rate (mu)", min_value=0.0, format="%0.2f", value=st.session_state.mu)
st.session_state.stand_by = st.selectbox("Standby mode", ["Warm", "Cold"], index=["Warm", "Cold"].index(st.session_state.stand_by))

if st.button("Calculate Uptime"):
    uptime = calculate_uptime(st.session_state.n, st.session_state.k, st.session_state.r, st.session_state.l, st.session_state.mu, st.session_state.stand_by)
    st.success(f"Estimated system uptime probability: {uptime:.6f}")

st.header("Optimal Cost Estimator")
st.session_state.cost_c = st.number_input("Cost per component (c)", min_value=0.0, format="%0.2f", value=st.session_state.cost_c)
st.session_state.cost_r = st.number_input("Cost per repairman (r)", min_value=0.0, format="%0.2f", value=st.session_state.cost_r)
st.session_state.cost_d = st.number_input("Cost of downtime (d)", min_value=0.0, format="%0.2f", value=st.session_state.cost_d)

if st.button("Estimate Optimal n and r"):
    best_combo, min_cost = estimate_n_r(st.session_state.k)
    st.success(f"Best configuration: number of components = {best_combo[0]}, and number of repairman = {best_combo[1]}")
    st.info(f"Estimated total cost: {min_cost:.2f}")

    uptime = calculate_uptime(best_combo[0], st.session_state.k, best_combo[1], st.session_state.l,
                              st.session_state.mu, st.session_state.stand_by)
    st.success(f"Estimated system uptime probability: {uptime:.6f}")

st.header("Cost Landscape Visualization")
if st.button("Show Cost Landscape"):
    visualize_cost_landscape(st.session_state.k, st.session_state.l,
                             st.session_state.mu, st.session_state.stand_by,
                             st.session_state.cost_c, st.session_state.cost_r,
                             st.session_state.cost_d)
