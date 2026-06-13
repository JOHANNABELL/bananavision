import cv2
import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler
from dataset.data_processing import clean_and_resize_images


def extract_banana_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    h, w = img.shape[:2]
    total_pixels_image = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([25, 255, 200])
    lower_dark = np.array([0, 40, 0])
    upper_dark = np.array([179, 255, 60])

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow), cv2.inRange(hsv, lower_green, upper_green))
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_brown, upper_brown))
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_dark, upper_dark))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    fruit_area = cv2.contourArea(main_contour)

    if fruit_area == 0:
        return None

    mask_fruit_filled = np.zeros_like(mask)
    cv2.drawContours(mask_fruit_filled, [main_contour], -1, 255, -1)

    b, g, r = cv2.split(img)

    mean_bgr, std_bgr = cv2.meanStdDev(img, mask=mask_fruit_filled)
    mean_hsv, _ = cv2.meanStdDev(hsv, mask=mask_fruit_filled)

    mean_B, mean_G, mean_R = mean_bgr.flatten()
    std_B, std_G, std_R = std_bgr.flatten()
    mean_H, mean_S, _ = mean_hsv.flatten()

    r_int = r.astype(np.int16)
    g_int = g.astype(np.int16)
    b_int = b.astype(np.int16)

    vert_cond = (g_int > r_int) & (g_int > b_int) & (g_int > 50) & (mask_fruit_filled > 0)
    pct_vert = (np.count_nonzero(vert_cond) / fruit_area)

    jaune_cond = (r_int > 100) & (g_int > 100) & (b_int < 100) & (mask_fruit_filled > 0)
    pct_jaune = (np.count_nonzero(jaune_cond) / fruit_area)

    lum = r_int + g_int + b_int
    sombre_cond = (lum < 150) & (mask_fruit_filled > 0)
    pct_sombre = (np.count_nonzero(sombre_cond) / fruit_area)

    blanc_cond = (lum > 660) & (mask_fruit_filled > 0)
    pct_blanc = (np.count_nonzero(blanc_cond) / fruit_area)

    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * math.pi * fruit_area) / (perimeter ** 2) if perimeter > 0 else 0

    aire_relative = fruit_area / total_pixels_image

    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = fruit_area / hull_area if hull_area > 0 else 0

    return {
        "moyenne_R": mean_R,
        "moyenne_V": mean_G,
        "moyenne_B": mean_B,
        "std_R": std_R,
        "std_V": std_G,
        "std_B": std_B,
        "pct_vert": pct_vert,
        "pct_jaune": pct_jaune,
        "pct_sombre": pct_sombre,
        "pct_blanc": pct_blanc,
        "moyenne_H": mean_H,
        "moyenne_S": mean_S,
        "circularite": circularity,
        "aire_relative": aire_relative,
        "solidite": solidity
    }


def process_train_folders(root_folder, label):
    data = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    print(f"Exploration de l'arborescence : '{root_folder}' ...")
    for root, dirs, files in os.walk(root_folder):
        if os.path.basename(root) == 'valid':
            print(f"-> Extraction en cours pour la classe [{label}] dans : {root}")
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_path = os.path.join(root, file)
                    try:
                        feats = extract_banana_features(image_path)
                        if feats is not None:
                            feats["classe"] = label
                            data.append(feats)
                    except Exception as e:
                        print(f"⚠️ Erreur sur {image_path} : {e}")
    return data


def build_dataset(path_export, path_rejet, output_file):
    data_export = process_train_folders(path_export, "export")
    print(f"Images export trouvées : {len(data_export)}")

    data_rejet = process_train_folders(path_rejet, "rejet")
    print(f"Images rejet trouvées : {len(data_rejet)}")

    all_data = data_export + data_rejet
    df = pd.DataFrame(all_data)

    if df.empty:
        print("Aucune donnée à sauvegarder. Vérifiez les chemins.")
        return None

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nApplication du StandardScaler...")
    X = df.drop(columns=['classe'])
    y = df['classe']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['classe'] = y.values

    df_scaled.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
    df_scaled.to_csv(output_file.replace('.parquet', '.csv'), index=False)

    print(f"\nDataset final sauvegardé : {output_file}")
    print(f"→ {len(df_scaled)} images au total, classes : {df_scaled['classe'].value_counts().to_dict()}")

    return df_scaled


if __name__ == "__main__":
    base_dataset_dir = "../../dataset"
    path_export = os.path.join(base_dataset_dir, "export")
    path_rejet = os.path.join(base_dataset_dir, "rejeter")
    output_parquet = os.path.join(base_dataset_dir, "features_bananes_validation.parquet")

    if os.path.exists(base_dataset_dir):
        clean_and_resize_images(base_dataset_dir, target_size=(224, 224))
    else:
        print(f"Erreur : Le dossier parent {base_dataset_dir} est introuvable.")
        exit()

    df_final = build_dataset(path_export, path_rejet, output_parquet)