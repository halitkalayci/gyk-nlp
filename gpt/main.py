import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = [
    "ali okula gitti",
    "veli sinemaya gitti",
    "ayşe markete gitti",
    "fatma işe gitti"
]

# 1. Tokenization (manual)

vocab = { "<PAD>": 0 } # <PAD> => Padding

index = 1

for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = index
            index += 1

#

# encode()
def encode(sentence):
    tokens = sentence.split()
    # cümlenin son kelimesi hariç tamamı - son kelimesi
    return torch.tensor([vocab[token] for token in tokens[:-1]], dtype=torch.long), torch.tensor(vocab[tokens[-1]], dtype=torch.long)
#

data = [encode(sentence) for sentence in sentences]

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Head -> Attentionda kaç farklı dikkat edilecek durum? (Odak noktası)
        # özne-fiil
        # sıfat-isim
        # özne-özne bağı
        # -------------------------- TAMAMEN TANIM --------------------------
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )
        # --------------------------
    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :] #pozisyon kadar slice

        # Query => Ne arıyorum? 
        # Key => Hangi bilgilerin keylerini karşılaştıracağım? 
        # Value => Sonuç olarak hangi bilgiyi çekeceğim?
        attn_out, _ = self.attn(x, x, x)

        out = self.ff(attn_out[:, -1, :])

        return out


model = MiniGPT(vocab_size=len(vocab), d_model=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    total_loss = 0
    for x, y in data:
        x = x.unsqueeze(0) #batch dimension
        # tahmin
        out = model(x)
        # gerçek etiketle karşılaştırma
        loss = loss_fn(out, y.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} Loss: {total_loss:.4f}")

# Predict fonksiyonunu kodlayalım.