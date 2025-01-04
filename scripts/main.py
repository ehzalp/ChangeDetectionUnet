from model import UNet  
from imports import *

# ONNX model path
onnx_model_path = '/Users/ehzalp/Desktop/ChangeDetection/Unet.onnx'
session = ort.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Image path 
image_path_A = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/images_A/79117_50995.jpg'  # img_A görselinizin yolu
image_path_B = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/images_B/79117_50995.jpg'  # img_B görselinizin yolu
label_mask_path = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/masks/79117_50995.jpg' 

img_A = Image.open(image_path_A).convert('RGB')
img_B = Image.open(image_path_B).convert('RGB')
label_mask = Image.open(label_mask_path).convert('1') 

# Görüntü ön işleme (modelinizin eğitiminde kullanılan işlemlerle aynı olmalıdır)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Modelinizin beklediği boyut (örneğin 256x256)
    transforms.ToTensor(),  # Görüntüyü Tensor'a dönüştür
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Modelinizin kullandığı normalizasyon
])

img_A_tensor = transform(img_A)
img_B_tensor = transform(img_B)

# img_A ve img_B'yi birleştirerek 6 kanal oluşturuyoruz
img_combined = torch.cat((img_A_tensor, img_B_tensor), dim=0)  # (6, 256, 256)

# Batch boyutu ekleyin
img_combined = img_combined.unsqueeze(0)  # (1, 6, 256, 256)

# NumPy array'e dönüştürün
input_array = img_combined.numpy()
# ONNX Runtime ile tahmin yapın
onnx_output = session.run([output_name], {input_name: input_array})[0]

# Sonuçları yazdırın
print("ONNX model çıktı şekli:", onnx_output.shape)

# Prediction mask: Çıktıyı threshold ile binary mask'e dönüştür
threshold = 0.5  # Binary mask için threshold değeri
prediction_mask = (onnx_output > threshold).astype(np.uint8)  # 0 ve 1 değerleri arasında dönüşüm
prediction_mask = prediction_mask.squeeze()  # Boyut (1, 1, 256, 256) -> (256, 256)

label_mask_np = np.array(label_mask)

# Görselleştirme
plt.figure(figsize=(12, 12))

# Before image (img_A)
plt.subplot(1, 4, 1)
plt.imshow(img_A)
plt.title("Before Image (A)")
plt.axis('off')

# After image (img_B)
plt.subplot(1, 4, 2)
plt.imshow(img_B)
plt.title("After Image (B)")
plt.axis('off')

# Label mask
plt.subplot(1, 4, 3)
plt.imshow(label_mask, cmap='gray')
plt.title("Label Mask")
plt.axis('off')

# Prediction mask
plt.subplot(1, 4, 4)
plt.imshow(prediction_mask, cmap='gray')
plt.title("Prediction Mask")
plt.axis('off')

# Göster
plt.tight_layout()
plt.show()