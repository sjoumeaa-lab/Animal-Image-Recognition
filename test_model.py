import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ---------- 1. DEFINE THE MODEL ARCHITECTURE (MUST MATCH YOUR train_cnn.py) ----------
class AnimalCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(AnimalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------- 2. SETUP DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 3. LOAD THE MODEL ----------
model_path = "models/cnn_model.pth"
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    print("Make sure you are running this script from the project root folder.")
    exit()

num_classes = 5   # CHANGE THIS if you have more/fewer classes
model = AnimalCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ---------- 4. DEFINE CLASS NAMES (MATCH YOUR DATASET FOLDER ORDER) ----------
# ⚠️ IMPORTANT: This order MUST match the order of folders in your dataset.
# If your train folders are alphabetically sorted, they are: cat, dog, elephant, horse, [your 5th]
# Adjust this list based on your actual dataset folders.
class_names = ['cat', 'dog', 'elephant', 'horse', 'other']   # <--- UPDATE THIS

# ---------- 5. PREPROCESSING (SAME AS TRAINING) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- 6. TEST ON AN IMAGE ----------
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    # Display result
    predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class {predicted_idx}"
    print(f"\n📸 Image: {image_path}")
    print(f"✅ Prediction: {predicted_label}")
    print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Show top 3 predictions
    top3 = torch.topk(probabilities, 3)
    print("\n🏆 Top 3 guesses:")
    for i in range(3):
        idx = top3.indices[i].item()
        name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        print(f"   {i+1}. {name}: {top3.values[i].item():.4f}")

# ---------- 7. RUN TESTS ----------
if __name__ == "__main__":
    # Test on cat image
    predict_image("test_cat.jpg")
    
    # Test on the other image if it exists
    if os.path.exists("test_animal.jpg"):
        predict_image("test_animal.jpg")