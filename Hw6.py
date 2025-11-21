import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


face_ref = plt.imread("face_reference.jpg")
face_input = plt.imread("face_input.jpg")




def compute_errors(img1, img2):
    mae = np.mean(np.abs(img1 - img2))
    mse = np.mean((img1 - img2) ** 2)
    return mae, mse

def to_grayscale(img):
    if img.ndim == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img




face_ref_gray = to_grayscale(face_ref)
face_input_gray = to_grayscale(face_input)

h, w = face_ref_gray.shape


def resize_numpy(img, new_h, new_w):
    h_old, w_old = img.shape
    row_ratio = h_old / new_h
    col_ratio = w_old / new_w
    resized = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            resized[i, j] = img[int(i * row_ratio), int(j * col_ratio)]
    return resized

face_input_resized = resize_numpy(face_input_gray, h, w)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(face_ref_gray, cmap='gray')
plt.title("Reference Face")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(face_input_gray, cmap='gray')
plt.title("Input Face")
plt.axis('off')

plt.show()


difference = np.abs(face_ref_gray - face_input_resized)
print("Mean difference:", np.mean(difference))


mae = np.mean(np.abs(face_ref_gray - face_input_resized))
print("Mean Absolute Difference:", mae)


mse = np.mean((face_ref_gray - face_input_resized) ** 2)
print("Mean Squared Difference:", mse)



threshold_mae = 20


if mae < threshold_mae:
    result_text = f"Face Matched (MAE={mae:.2f})"
else:
    result_text = f"Face Not Matched (MAE={mae:.2f})"


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(face_ref_gray, cmap='gray')
plt.title("Reference Face")

plt.subplot(1,2,2)
plt.imshow(face_input_resized, cmap='gray')
plt.title(result_text)

plt.show()






input_images = ["face_input.jpg", "face_input.jpg", "face_input.jpg"]


results = []

for img_path in input_images:
    img = plt.imread(img_path)
    img_gray = to_grayscale(img)
    img_resized = resize_numpy(img_gray, h, w)

    mae, mse_val = compute_errors(face_ref_gray, img_resized)
    results.append({"image": img_path, "MAE": mae, "MSE": mse_val})


df_results = pd.DataFrame(results)
print(df_results)


plt.figure(figsize=(8,5))
plt.bar(df_results['image'], df_results['MSE'], color='skyblue')
plt.xlabel("Input Image")
plt.ylabel("MSE with Reference")
plt.title("MSE Comparison of Input Images")
plt.xticks(rotation=45)
plt.show()