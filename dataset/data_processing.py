import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def clean_and_resize_images(dataset_dir, target_size):
    print(f"--- Nettoyage et redimensionnement du dataset ---")
    removed_count = 0
    resized_count = 0
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()

                    with Image.open(file_path) as img:
                        img.load()

                        img_rgb = img.convert('RGB')

                        img_resized = img_rgb.resize(target_size, Image.Resampling.LANCZOS)

                        img_resized.save(file_path)
                        resized_count += 1

                except Exception as e:
                    print(f"-> Image corrompue supprimée : {file_path} (Erreur: {e})")
                    os.remove(file_path)
                    removed_count += 1

    print(f"--- Terminé ---\n")


def get_dataset_loaders(dataset_parent_dir, batch_size=32, num_workers=4):
    print(f"--- Configuration des pipelines et des DataLoaders ---")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    dataloaders = {}
    class_names = None

    for folder_name in os.listdir(dataset_parent_dir):
        folder_path = os.path.join(dataset_parent_dir, folder_name)

        if os.path.isdir(folder_path):
            if folder_name.lower() == 'train':
                print(f"-> Dossier '{folder_name}' détecté : Application du pipeline d'AUGMENTATION.")
                current_transform = train_transforms
                shuffle_setting = True
            else:
                print(f"-> Dossier '{folder_name}' détecté : Application du pipeline d'ÉVALUATION standard.")
                current_transform = eval_transforms
                shuffle_setting = False

            dataset = ImageFolder(root=folder_path, transform=current_transform)

            if folder_name.lower() == 'train':
                class_names = dataset.classes

            dataloaders[folder_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_setting,
                num_workers=num_workers,
                pin_memory=True
            )
            print(f"   Moteur DataLoader prêt pour '{folder_name}' ({len(dataset)} images trouvées).")

    print(f"--- Configuration terminée. ---\n")
    return dataloaders, class_names