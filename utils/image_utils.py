import os
import cv2
import shutil

def cache_resized_images(input_dir, output_dir, size=(224, 224)):

    """
    Tüm raw görselleri okur, yeniden boyutlandirir ve processed klasörüne kaydeder.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    class_names = sorted(os.listdir(input_dir))
    for class_name in class_names:
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for fname in os.listdir(class_input_path):
            if fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                in_path = os.path.join(class_input_path, fname)
                out_path = os.path.join(class_output_path, fname)

                img = cv2.imread(in_path)
                if img is None:
                    continue
                resized = cv2.resize(img, size)
                cv2.imwrite(out_path, resized)

if __name__ == "__main__":
    # Test çağrısı (örnek)
    cache_resized_images("data/raw", "data/processed")


