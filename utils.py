import pandas as pd

def sample_dataset(df, total_samples=100):
    # Total samples required
    #total_samples = 100

    # Group by poses and angles
    grouped = df.groupby(['pose', 'deg'])

    # Calculate the number of groups
    num_groups = len(grouped)

    # Calculate samples per group (approximately)
    samples_per_group = total_samples // num_groups

    # Collect samples from each group
    samples = []

    for _, group in grouped:
        # Determine the number of samples to take from this group
        n_samples = min(samples_per_group, len(group))
        
        # Randomly sample from the group
        sampled_group = group.sample(n_samples)
        
        samples.append(sampled_group)

    # Concatenate all samples into a single DataFrame
    selected_samples = pd.concat(samples)

    # If we have fewer than 100 samples due to rounding, sample additional rows randomly from the result
    if len(selected_samples) < total_samples:
        additional_samples = df.sample(total_samples - len(selected_samples))
        selected_samples = pd.concat([selected_samples, additional_samples])

    # If we have more than 100 samples, randomly sample 100 from the result
    if len(selected_samples) > total_samples:
        selected_samples = selected_samples.sample(total_samples)

    # Get the rest of the dataset by dropping the selected indices from the original DataFrame
    remaining_samples = df.drop(selected_samples.index)

    # Reset index for final result
    selected_samples = selected_samples.reset_index(drop=True)
    remaining_samples = remaining_samples.reset_index(drop=True)

    return selected_samples, remaining_samples
