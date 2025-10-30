# 🧠 Proof of Concept: Image Captioning

Ovaj projekat predstavlja **proof of concept** implementaciju modela za *image captioning* — generisanje opisa slike pomoću dubokog učenja.

Svi eksperimenti, modeli, i pomoćne skripte se nalaze u folderu **`Proof-of-concept`**, koji sadrži sav relevantan kod za treniranje i testiranje modela.

---

## 🚀 Pokretanje treninga

Za pokretanje treninga modela koristiti sledeće komande u terminalu (Windows CMD):

```cmd
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
python main.py
```

Ove promenljive okruženja služe za isključivanje TensorFlow informativnih logova i upozorenja kako bi izlaz u konzoli bio čitljiviji.

---

## 🖼️ Pokretanje demo aplikacije

Nakon što je model istreniran, moguće je testirati generisanje opisa slike pomoću demo aplikacije:

```cmd
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
python demo.py putanja/do/slike.jpg
```

---

## 📄 Tehnički izveštaj

U okviru projekta se nalazi i PDF dokument pod nazivom
**`Tehnički izveštaj - Tehnološki projekat.pdf`**,
koji sadrži sledeće informacije:

* opis korišćenih tehnologija,
* prikaz arhitekture softvera u pogledu glavnih komponenti,
* najznačajnije detalje implementacije,
* pregled i analizu ostvarenih rezultata.

---

Hoćeš da ti ovo formatiram kao gotov `README.md` fajl za preuzimanje (sa emoji oznakama i Markdown formatiranjem zadržanim)?
