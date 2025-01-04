from imports import *
from train_Unet import model, dataloaders, A_folder, B_folder

# inference model to get change detection mask
def inference(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for inputs_combined, label in dataloader:
            inputs_combined = inputs_combined.to(device)
            label = label.to(device)
            output = model(inputs_combined)
            outputs.append(output.cpu())
            labels.append(label.cpu())
    return torch.cat(outputs), torch.cat(labels)


def visualize_results(img_A_path, img_B_path, pred_mask, label_mask):
    """
    Çıktıları görselleştirme: 
    - Önceki görüntü (Before)
    - Sonraki görüntü (After)
    - Tahmin maskesi ve Gerçek maskesi
    """
    # Eşikleme uygulama
    threshold = 0.5
    binary_mask = (pred_mask.cpu().numpy() > threshold).astype(np.uint8)  # 0 ve 1 arasında eşik uygular
    
    # Görselleştirme için bir grid oluştur
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 satır, 4 sütun
    
    # Before (A Image) - Load image from path
    img_A = Image.open(img_A_path).convert('RGB')
    img_A = np.array(img_A)
    axes[0].imshow(img_A)  # RGB görüntüyü göster
    axes[0].set_title("Before (A Image)")
    axes[0].axis('off')
    
    # After (B Image) - Load image from path
    img_B = Image.open(img_B_path).convert('RGB')
    img_B = np.array(img_B)
    axes[1].imshow(img_B)  # RGB görüntüyü göster
    axes[1].set_title("After (B Image)")
    axes[1].axis('off')
    
    # Tahmin Maskesi (Prediction)
    axes[2].imshow(binary_mask.squeeze(), cmap='gray')  # Binary mask için gri tonlama
    axes[2].set_title("Prediction Mask")
    axes[2].axis('off')
    
    # Gerçek Maskesi (Label)
    axes[3].imshow(label_mask.squeeze(), cmap='gray')  # Binary mask için gri tonlama
    axes[3].set_title("Label Mask")
    axes[3].axis('off')

    # Grafikleri göster
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #eval model
    model.load_state_dict(torch.load('/Users/ehzalp/Desktop/ChangeDetection/best_model_unet_50ep.pth', map_location=torch.device('cpu')))
    model.eval()
    # Place the code that runs when the script is executed directly here
    pred_mask, label_mask = inference(model, dataloaders['test'])
    print(pred_mask, label_mask)
    # Get a list of image paths from your folders
    A_image_paths = [os.path.join(A_folder, filename) for filename in os.listdir(A_folder) if filename.endswith('.jpg')]
    B_image_paths = [os.path.join(B_folder, filename) for filename in os.listdir(B_folder) if filename.endswith('.jpg')]
    # Visualize results for the first image pair
    visualize_results(A_image_paths[0], B_image_paths[0], pred_mask[0], label_mask[0])



