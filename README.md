# 🧠 Proof of Concept: Image Captioning

Ovaj projekat predstavlja **proof of concept** implementaciju modela za *image captioning* — generisanje opisa slike pomoću dubokog učenja.  

---

## 🚀 Pokretanje treninga

Za pokretanje treninga modela koristi sledeće komande u terminalu (Windows CMD):

```cmd
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
python main.py
````

Ove promenljive okruženja služe za isključivanje TensorFlow informativnih logova i upozorenja kako bi izlaz u konzoli bio čitljiviji.

---

## 🖼️ Pokretanje demo aplikacije

Nakon što je model istreniran, možeš testirati generisanje opisa slike pomoću demo aplikacije:

```cmd
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
python demo.py putanja/do/slike.jpg
```
