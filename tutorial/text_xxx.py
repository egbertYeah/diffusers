from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, filters, morphology

# Load the marked image
marked_image_path = "datasets/zaohe/tmp/09-45-41-357_1.jpg"
marked_image = Image.open(marked_image_path).convert("L")  # Convert to grayscale

# Convert image to numpy array
marked_image_array = np.array(marked_image)

# Apply threshold to binarize the image
# Adjust thresholding based on red-marked areas
threshold_value = filters.threshold_otsu(marked_image_array)  # Use Otsu's method for threshold
binary_marked_image = marked_image_array < threshold_value  # Invert due to grayscale nature

# Remove small objects (noise) using morphological operations
cleaned_image = morphology.remove_small_objects(binary_marked_image, min_size=100)

# Label connected components
labeled_image, num_features = ndimage.label(cleaned_image)

# Plot the image and show the labeled regions
plt.figure(figsize=(6, 6))
plt.imshow(labeled_image, cmap='nipy_spectral')
# Extract properties of labeled regions
regions = measure.regionprops(labeled_image)
# Analyze the shape properties of the detected regions to differentiate between "complete" and "incomplete" half dates

# Criteria for completeness (you might need to adjust these thresholds based on actual image characteristics)
complete_count = 0
incomplete_count = 0

# Define thresholds for "complete" and "incomplete" classification
# These values are arbitrary and should be fine-tuned based on domain knowledge or additional samples
area_threshold = 500  # Minimum area to be considered "complete"
eccentricity_threshold = 0.8  # Maximum eccentricity to be considered "complete" (values closer to 0 indicate circular shapes)

# Iterate over all detected regions to classify them
for region in regions:
    # Get the properties of each region
    area = region.area
    eccentricity = region.eccentricity  # Measure how elongated the shape is

    # Classify based on area and eccentricity
    if area >= area_threshold and eccentricity < eccentricity_threshold:
        complete_count += 1
    else:
        incomplete_count += 1

# Iterate over the detected regions and plot their bounding boxes
for region in regions:
    # Get bounding box coordinates (min_row, min_col, max_row, max_col)
    minr, minc, maxr, maxc = region.bbox
    area = region.area
    eccentricity = region.eccentricity

    # Classify as "complete" or "incomplete" based on the previous criteria
    if area >= area_threshold and eccentricity < eccentricity_threshold:
        # Draw green rectangle for complete half dates
        plt.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], color='green', linewidth=2)
    else:
        # Draw red rectangle for incomplete half dates
        plt.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], color='red', linewidth=2)



plt.title(f"Detected potential 'half dates': {num_features}")
plt.axis('off')
plt.savefig("res.png")
# Output the number of detected regions that potentially match the marked areas
print(num_features)
