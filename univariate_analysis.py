import numpy as np
import matplotlib.pyplot as plt


def compute_required_data():
    with open('CategoryLabels.txt', 'r') as f:
        category_labels = [line.split('"')[3] for line in f.readlines()[1:]]

    category_vectors = np.loadtxt('CategoryVectors.txt', delimiter=',', skiprows=1)
    neural_responses = np.loadtxt('NeuralResponses_S1.txt', delimiter=',', skiprows=1)

    animate_index = category_labels.index('animate')
    inanimate_index = category_labels.index('inanim')

    animate_vectors = category_vectors[:, animate_index]
    inanimate_vectors = category_vectors[:, inanimate_index]

    anim_voxels = neural_responses[animate_vectors == 1]
    inanim_voxels = neural_responses[inanimate_vectors == 1]

    avg_voxel_values_anim = np.mean(anim_voxels, axis=1)
    avg_voxel_values_inanim = np.mean(inanim_voxels, axis=1)

    return anim_voxels, inanim_voxels, avg_voxel_values_anim, avg_voxel_values_inanim


def paired_t_test(vectors_1, vectors_2):
    differences = np.array(vectors_1) - np.array(vectors_2)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(vectors_1)
    t_value = mean_diff / (std_diff / np.sqrt(n))
    return t_value


def get_voxel_average_difference(animate_images, inanimate_images):
    animate_averages = np.mean(animate_images, axis=0)
    inanimate_averages = np.mean(inanimate_images, axis=0)
    voxel_differences = animate_averages - inanimate_averages
    return voxel_differences


def plot_average_response_amplitude(avg_vox_vals_animate, avg_vox_vals_inanimate):
    fig, ax = plt.subplots(figsize=(4.5, 7))

    sem_animate = np.std(avg_vox_vals_animate) / np.sqrt(len(avg_vox_vals_animate))
    sem_inanimate = np.std(avg_vox_vals_inanimate) / np.sqrt(len(avg_vox_vals_inanimate))

    bar_width = 0.015
    bar_gap = 0.01

    _ = ax.bar(
        0.4,
        np.mean(avg_vox_vals_animate),
        bar_width,
        color='b',
        yerr=sem_animate,
        ecolor='red',
        edgecolor='black',
        linewidth=1.2,
        label='Animate',
        capsize=5,)

    _ = ax.bar(
        0.4 + bar_width + bar_gap,
        np.mean(avg_vox_vals_inanimate),
        bar_width,
        color='g',
        yerr=sem_inanimate,
        ecolor='red',
        edgecolor='black',
        linewidth=1.2,
        label='Inanimate',
        capsize=5,)

    ax.set_ylabel('Response Amplitude')
    ax.set_title('Average Response Amplitude')
    ax.set_xticks([0.4, 0.4 + bar_width + bar_gap])
    ax.set_xticklabels(['Animate', 'Inanimate'])
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_voxel_average_difference(vox_diff):
    figure, axes = plt.subplots(figsize=(6, 10))
    bar_width = 1
    bar_positions = np.arange(20)

    axes.bar(
        bar_positions,
        vox_diff[:20],
        bar_width,
        color='orange',
        edgecolor='black',
        linewidth=1.2,
        label='Voxel Differences')

    axes.set_xlabel('Voxel')
    axes.set_ylabel('Response Amplitude')
    axes.set_title('Animate minus Inanimate')
    axes.legend()

    plt.show()


animate_voxels, inanimate_voxels, average_voxel_values_animate, average_voxel_values_inanimate = compute_required_data()

print("Voxel Values for Animate Objects:")
print(animate_voxels)
print("\nVoxel Values for Inanimate Objects:")
print(inanimate_voxels)
print("\nAverage Voxel Values for Animate Objects:")
print(average_voxel_values_animate)
print(len(average_voxel_values_animate))
print("\nAverage Voxel Values for Inanimate Objects:")
print(average_voxel_values_inanimate)
print(len(average_voxel_values_inanimate))

plot_average_response_amplitude(average_voxel_values_animate, average_voxel_values_inanimate)

t_stat = paired_t_test(average_voxel_values_animate, average_voxel_values_inanimate)

print("\nt-value:", t_stat)

voxel_diff = get_voxel_average_difference(animate_voxels, inanimate_voxels)

print('\nDifference between average animate / inanimate voxel:')
print(voxel_diff)
print(len(voxel_diff))

plot_voxel_average_difference(voxel_diff)
