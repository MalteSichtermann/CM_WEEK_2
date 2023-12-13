import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# For Task 1 & 2:
def compute_required_data(sample_file):
    with open('CategoryLabels.txt', 'r') as f:
        category_labels = [line.split('"')[3] for line in f.readlines()[1:]]

    category_vectors = np.loadtxt('CategoryVectors.txt', delimiter=',', skiprows=1)
    neural_responses = np.loadtxt(sample_file, delimiter=',', skiprows=1)

    # Task 1
    animate_index = category_labels.index('animate')
    inanimate_index = category_labels.index('inanim')
    # Task 2
    human_index = category_labels.index('human')
    non_human_index = category_labels.index('nonhuman')

    # Task 1
    animate_vectors = category_vectors[:, animate_index]
    inanimate_vectors = category_vectors[:, inanimate_index]
    # Task 2
    human_vectors = category_vectors[:, human_index]
    nonhuman_vectors = category_vectors[:, non_human_index]

    # Task 1
    anim_voxels = neural_responses[animate_vectors == 1]
    inanim_voxels = neural_responses[inanimate_vectors == 1]
    # Task 2
    hum_voxels = neural_responses[human_vectors == 1]
    nonhum_voxels = neural_responses[nonhuman_vectors == 1]
    hum_voxels = hum_voxels[:-4, :]

    avg_voxel_values_anim = np.mean(anim_voxels, axis=1)
    avg_voxel_values_inanim = np.mean(inanim_voxels, axis=1)

    return anim_voxels, inanim_voxels, avg_voxel_values_anim, avg_voxel_values_inanim, hum_voxels, nonhum_voxels


# For Task 1:
def paired_t_test(vectors_1, vectors_2):
    differences = np.array(vectors_1) - np.array(vectors_2)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(vectors_1)
    t_value = mean_diff / (std_diff / np.sqrt(n))
    return t_value


# For Task 1 & 2:
def get_voxel_average_difference(animate_images, inanimate_images):
    animate_averages = np.mean(animate_images, axis=0)
    inanimate_averages = np.mean(inanimate_images, axis=0)
    voxel_differences = animate_averages - inanimate_averages
    return voxel_differences


# For Task 1:
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


# For Task 1:
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


# For Task 2:
def train_and_predict(data_set_1, data_set_2, sample_size, amount_weights):
    training_data_set_1 = data_set_1[:sample_size, :]
    training_data_set_2 = data_set_2[:sample_size, :]
    test_data_set_1 = data_set_1[sample_size:, :]
    test_data_set_2 = data_set_2[sample_size:, :]

    labels = np.concatenate([np.ones(sample_size), -np.ones(sample_size)])
    print(f'Actual Labels:\n{labels}')

    training_data = np.concatenate([training_data_set_1, training_data_set_2])
    test_data = np.concatenate([test_data_set_1, test_data_set_2])

    clf = svm.SVC(kernel='linear')
    clf.fit(training_data, labels)

    predictions = clf.predict(test_data)
    accuracy = np.mean(predictions == labels)
    svm_weights = clf.coef_[0][:amount_weights]

    return predictions, accuracy, svm_weights


# For Task 2:
def plot_svm_weights_vs_voxel_diff(svm_weights, vox_diff, sample_size, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(svm_weights, vox_diff[:sample_size], color='white', edgecolors='blue')
    plt.xlabel('SVM Weights')
    plt.ylabel('Voxel Differences')
    plt.title(f'SVM Weights vs. Voxel Differences - {title}')
    plt.show()


# Task 1 - Main:
animate_voxels, inanimate_voxels, average_voxel_values_animate, average_voxel_values_inanimate, _, _ = (
    compute_required_data('NeuralResponses_S1.txt')
)
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


# Task 2 - Main:
animate_voxels2, inanimate_voxels2, _, _, human_voxels, nonhuman_voxels = (
    compute_required_data('NeuralResponses_S2.txt')
)
print("\nVoxel Values for Animate Objects Task 2:")
print(animate_voxels2)
print("\nVoxel Values for Inanimate Objects Task 2:")
print(inanimate_voxels2)

predictions_anim, accuracy_anim, weights_anim = (
    train_and_predict(animate_voxels2, inanimate_voxels2, 22, 20)
)
print("\nPredictions Anim:")
print(predictions_anim)
print("\nAccuracy Anim:")
print(accuracy_anim)

voxel_diff_anim = get_voxel_average_difference(animate_voxels2, inanimate_voxels2)
plot_svm_weights_vs_voxel_diff(weights_anim, voxel_diff_anim, 20, 'Animate/Inanimate')

correlation_matrix_anim = np.corrcoef(weights_anim, voxel_diff_anim[:20])
print(f'\nCorrelation matrix (r) Animate-Inanimate: {correlation_matrix_anim}')

pearson_r_anim = correlation_matrix_anim[0, 1]
print(f'\nPearson correlation coefficient r Animate-Inanimate: {pearson_r_anim}')

# Use the functions for the Human-Nonhuman task
predictions_human, accuracy_human, weights_human = (
    train_and_predict(human_voxels, nonhuman_voxels, 10, 10)
)
print("\nPredictions Human:")
print(predictions_human)
print("\nAccuracy Human:")
print(accuracy_human)

voxel_diff_human = get_voxel_average_difference(human_voxels, nonhuman_voxels)
plot_svm_weights_vs_voxel_diff(weights_human, voxel_diff_human, 10, 'Human/Nonhuman')

correlation_matrix_human = np.corrcoef(weights_human, voxel_diff_human[:10])
print(f'\nCorrelation matrix (r) Human-Nonhuman: {correlation_matrix_human}')

pearson_r_human = correlation_matrix_human[0, 1]
print(f'\nPearson correlation coefficient r Human-Nonhuman: {pearson_r_human}')
