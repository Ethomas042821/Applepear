import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# def plot_contribution_heatmap(model, activations):
#     # Dense(128) activations
#     dense_128_activations = activations[6][0]  # Shape: (128,)

#     # Weights from Dense(128) → Dense(2)
#     weights = model.layers[7].get_weights()[0]  # Shape: (128, 2)

#     # Compute contribution matrix: (neurons x classes)
#     contributions = dense_128_activations[:, np.newaxis] * weights  # Shape: (128, 2)

#     # Normalize for display (optional)
#     norm_contributions = contributions / np.max(np.abs(contributions))

#     # Transpose to make shape (2, 128) for y=classes, x=neurons
#     transposed = norm_contributions.T

#     # Create the heatmap
#     fig, ax = plt.subplots(figsize=(12, 3))  # Wider for horizontal layout
#     sns.heatmap(
#         transposed,
#         cmap="coolwarm",
#         center=0,
#         xticklabels=[f"N{i}" for i in range(contributions.shape[0])],
#         yticklabels=["Apple", "Pear"],
#         ax=ax
#     )
#     ax.set_title("Output Class Contributions (activation*corresponding weight) by dense Neurons")
#     ax.set_ylabel("Output Class")
#     ax.set_xlabel("Dense(128) Neurons")
#     st.pyplot(fig)

def plot_contribution_heatmap(model, activations):
    # Dense(128) activations
    dense_128_activations = activations[6][0]  # Shape: (128,)

    # Weights from Dense(128) → Dense(2)
    weights = model.layers[7].get_weights()[0]  # Shape: (128, 2)
    biases = model.layers[7].get_weights()[1]  # Shape: (2,)

    # Contribution matrix: (128 neurons, 2 classes)
    contributions = dense_128_activations[:, np.newaxis] * weights  # Shape: (128, 2)

    weighted_sum_apple = np.sum(dense_128_activations * weights[:, 0])
    weighted_sum_pear = np.sum(dense_128_activations * weights[:, 1])

    logits_apple = weighted_sum_apple + biases[0]
    logits_pear = weighted_sum_pear + biases[1]

    # Normalize and transpose the contribution matrix
    norm_contributions = contributions / np.max(np.abs(contributions))
    norm_contributions = norm_contributions.T  # Now shape is (2, 128)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 3))

    heatmap = sns.heatmap(
        norm_contributions,
        cmap="coolwarm",
        center=0,
        # xticklabels=[f"N{i}" for i in range(contributions.shape[0])],
        xticklabels=True,  # still generate ticks
        yticklabels=["Apple", "Pear"],
        ax=ax
    )

    ax.set_title(f"Weight*Activation Heatmap\n"
                 f"Logit = Weighted Sum + Bias\n"
                 f"Logit (Apple) = {logits_apple:.2f} | "
                 f"Logit (Pear) = {logits_pear:.2f}",
                fontsize=9)
    
    # ax.text(0.98, 0.90, f"Bias (Apple): {biases[0]:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7, bbox=dict(facecolor='white', boxstyle='round,pad=0.5'))
    # ax.text(0.98, 0.79, f"Bias (Pear): {biases[1]:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7, bbox=dict(facecolor='white', boxstyle='round,pad=0.5'))
    # ax.text(0.80, 0.90, f"Weighted Sum (Apple): {weighted_sum_apple:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7, bbox=dict(facecolor='white', boxstyle='round,pad=0.5'))
    # ax.text(0.80, 0.79, f"Weighted Sum (Pear): {weighted_sum_pear:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7, bbox=dict(facecolor='white', boxstyle='round,pad=0.5'))

    ax.set_xlabel("Dense(128) Neurons", fontsize=8)
    ax.set_ylabel("Output Class",fontsize=8)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4)
    ax.set_xticklabels([])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    # Resize the colorbar tick labels
    # Adjust colorbar tick label size
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=8) 


    st.pyplot(fig)