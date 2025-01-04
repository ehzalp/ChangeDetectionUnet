import torch
import torch.onnx
import numpy as np
import time
import copy
from model import UNet  
from imports import *  
from train_Unet import model

# Modeli yükleyin (önceden eğitilmiş bir modelin state_dict'ini yükleyebilirsiniz)
def load_model(model_path):
    model = UNet(n_classes=1)  # Modelinizi uygun şekilde yapılandırın
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

# Modeli ONNX formatına dönüştürme
def convert_model_to_onnx(model, output_path, example_input):
    model.eval()  # Modeli eval moduna alıyoruz
    torch.onnx.export(
        model,                     # Eğitilmiş model
        example_input,              # Modelin beklediği örnek giriş verisi
        output_path,               # ONNX formatında kaydedilecek dosyanın yolu
        export_params=True,        # Ağırlıkları ONNX dosyasına dahil et
        opset_version=11,          # Opset sürümü (ONNX sürümüyle uyumlu)
        do_constant_folding=True,  # Sabit katmanları katlamak için
        input_names=['input'],     # Giriş tensörünün adı
        output_names=['output'],   # Çıktı tensörünün adı
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dinamik batch boyutu
    )
    print(f"Model başarıyla {output_path} olarak kaydedildi.")

# Modeli doğrulama ve test etme
def validate_onnx_model(onnx_model_path, example_input):
    import onnx
    import onnxruntime as ort

    # ONNX modelini yükle
    onnx_model = onnx.load(onnx_model_path)

    # Modelin geçerliliğini kontrol et
    onnx.checker.check_model(onnx_model)
    print("ONNX model geçerli!")

    # ONNX runtime ile modeli çalıştır
    ort_session = ort.InferenceSession(onnx_model_path)

    # Örnek giriş verisini NumPy formatına dönüştür
    ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}

    # Modelin tahminini yap
    ort_outs = ort_session.run(None, ort_inputs)

    # Çıktıyı yazdır
    print("ONNX Çıktısı:", ort_outs[0])

def main():
    # Eğitilmiş modelin yolu
    model_path = '/Users/ehzalp/Desktop/ChangeDetection/best_model_unet_50ep.pth'  # Model dosyanızın yolu
    onnx_output_path = '/Users/ehzalp/Desktop/ChangeDetection/Unet.onnx'  # Dönüştürülen ONNX modelinin kaydedileceği dosya

    # Modeli yükle
    model = load_model(model_path)


    # Örnek bir giriş verisi oluştur (giriş boyutları modelin beklediği şekilde ayarlanmalı)
    example_input = torch.randn(1, 6, 256, 256)  # 1 örnek, 6 kanal (img_A ve img_B için 3 kanal her biri), 256x256 boyut

    # Modeli ONNX formatına dönüştür
    convert_model_to_onnx(model, onnx_output_path, example_input)

    # ONNX modelini test et
    validate_onnx_model(onnx_output_path, example_input)

if __name__ == '__main__':
    main()