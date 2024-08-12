import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import os

# Step 1: Load the image
image_path = 'Giraffe.jpeg'  # Replace with your image path
image = Image.open(image_path).convert('L')  # Convert image to grayscale
image_array = np.array(image)

# Step 2: Display the original image
plt.imshow(image_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# Step 3: Reshape the image to 2D array
m, n = image_array.shape
image_reshaped = image_array.reshape(m, n)

# Step 4: Apply SVD for Image Compression
k = 10  # Number of singular values to keep
svd = TruncatedSVD(n_components=k)
U = svd.fit_transform(image_reshaped)
Sigma = svd.singular_values_
VT = svd.components_

# Reconstruct the compressed image
compressed_image = np.dot(U, VT)

# Step 5: Display the compressed image
plt.imshow(compressed_image, cmap='gray')
plt.title(f"Compressed Image with k={k}")
plt.axis('off')
plt.show()

# Step 6: Calculate compression statistics
# Original data size
original_data_points = m * n
# Compressed data size
compressed_data_points = U.size + Sigma.size + VT.size

# Compression ratio
compression_ratio = original_data_points / compressed_data_points

# Step 7: Compare file sizes
original_size = os.path.getsize(image_path)  # Size of original image file in bytes
compressed_image_pil = Image.fromarray(compressed_image.astype(np.uint8))
compressed_image_path = 'compressed_image_k10.jpg'
compressed_image_pil.save(compressed_image_path)
compressed_size = os.path.getsize(compressed_image_path)  # Size of compressed image file in bytes

# Step 8: Print out the statistics
print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Compressed Image Size with k={k}: {compressed_size / 1024:.2f} KB")
print(f"Original Data Points: {original_data_points}")
print(f"Compressed Data Points: {compressed_data_points}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Original Dimensions: {m} x {n}")
print(f"Reconstructed Dimensions: {compressed_image.shape[0]} x {compressed_image.shape[1]}")

# Step 9: Plot the Singular Values
plt.plot(Sigma, 'bo-', label="Singular Values")
plt.title("Singular Values of the Image")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.legend()
plt.show()
