from imports import *

# Paths to the dataset
DATASET_DIR = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData'
IMAGE_SHAPE = (256, 256, 3)

# Load and preprocess data
def load_data(dataset_dir, image_shape):
    before_images, after_images, masks = [], [], []

    before_dir = os.path.join(dataset_dir, 'images_A')
    after_dir = os.path.join(dataset_dir, 'images_B')
    mask_dir = os.path.join(dataset_dir, 'masks')

    for filename in tqdm(os.listdir(before_dir)):
        if not filename.endswith('.jpg'):
            continue
        before_img = tf.keras.utils.load_img(os.path.join(before_dir, filename), target_size=image_shape[:2])
        after_img = tf.keras.utils.load_img(os.path.join(after_dir, filename), target_size=image_shape[:2])
        mask_img = tf.keras.utils.load_img(os.path.join(mask_dir, filename), target_size=image_shape[:2], color_mode='grayscale')

        before_images.append(np.array(before_img) / 255.0)
        after_images.append(np.array(after_img) / 255.0)
        masks.append(np.array(mask_img) / 255.0)

    before_images = np.array(before_images, dtype=np.float32)
    after_images = np.array(after_images, dtype=np.float32)
    masks = np.expand_dims(np.array(masks, dtype=np.float32), axis=-1)

    return before_images, after_images, masks


before, after, masks = load_data(DATASET_DIR, IMAGE_SHAPE)

# Split data into train and validation sets
train_before, val_before, train_after, val_after, train_masks, val_masks = train_test_split(
    before, after, masks, test_size=0.2, random_state=42
)


########################## DATA LOADER #############################

class ChangeDetectionDataset(Dataset):
    def __init__(self, A_folder, B_folder, labels_folder, transform=None):
        self.A_folder = A_folder
        self.B_folder = B_folder
        self.labels_folder = labels_folder
        self.transform = transform

        self.file_names = os.listdir(A_folder)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        if self.file_names[idx].endswith('.jpg'):
            file_name = self.file_names[idx]
        else:
            return self.__getitem__(idx + 1)  # Recursively try the next index

        img_A = Image.open(os.path.join(self.A_folder, file_name)).convert('RGB')  # Ensure RGB channels
        img_B = Image.open(os.path.join(self.B_folder, file_name)).convert('RGB')
        label = Image.open(os.path.join(self.labels_folder, file_name)).convert('L')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            label = self.transform(label)

        # Concatenate img_A and img_B along the channel dimension (dim=0 for batch)
        img_combined = torch.cat((img_A, img_B), dim=0)

        return img_combined, label

def get_data_loaders(A_folder, B_folder, labels_folder, batch_size=25, shuffle=True, num_workers=0):
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = ChangeDetectionDataset(A_folder, B_folder, labels_folder, transform=trans)
    val_set = ChangeDetectionDataset(A_folder, B_folder, labels_folder, transform=trans)
    test_set = ChangeDetectionDataset(A_folder, B_folder, labels_folder, transform=trans)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders
