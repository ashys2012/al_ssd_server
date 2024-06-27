    

import random
total_images = 10

num_images_to_label = 5


label_indices = set(random.sample(range(total_images), num_images_to_label))
all_indices = set(range(total_images))

# Find remaining indices as a set
remaining_indices = all_indices - label_indices

print(f"Label indices: {label_indices}")
print(f"Remaining indices: {remaining_indices}")