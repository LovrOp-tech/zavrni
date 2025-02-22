import os
import random
from robodk import robolink, robomath
from PIL import Image, ImageDraw, ImageFont
from datasets import DatasetDict, Dataset, Features, Image as HFImage, Value, Sequence

# Inicijalizacija RoboDK-a
RDK = robolink.Robolink()

# Definiranje raspona za postavljanje objekata
x_min, x_max = 150, 860
y_min, y_max = -200, 200
z = 0  # Visina (z koordinata) za postavljanje objekata

# Minimalna udaljenost između objekata
min_udaljenost = 141.42

# Lista imena objekata
imena_objekata = [f"{i}dio" for i in range(1, 11)]

# Direktorij za spremanje slika
direktorij = r"D:\\crna_podloga_250mmm"  

# Hugging Face token
HF_TOKEN = "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"  

# Postavljanje fonta 
velicina_fonta = 15
font = ImageFont.truetype("arial.ttf", velicina_fonta)


# Postavke kamere
pozicija_kamere = [475, 0, 250]
orijentacija_kamere = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  
focal_length_mm = 8
pixel_size_um = 16.4
sensor_width_pixels = 1920
sensor_height_pixels = 1080
focal_length_pixels = (focal_length_mm / pixel_size_um) * 1000 

# Priprema dataseta za Hugging Face
data = {"image_id": [], "image": [], "width": [], "height": [], "objects": []}


# ID counter za bounding bbox
ID_counter = 1

# Glavna for petlja za generiranje slika
for run in range(1, 11):
    # Učitavanje objekata iz RoboDK
    objekti = []   
    for ime in imena_objekata:
        obj = RDK.Item(ime) 
        objekti.append(obj)  

    postavljeni_objekti = []  
    globalne_koordinate = {}  

    for i, obj in enumerate(objekti, start=1):  
        valjana_pozicija = False
        pokusaji = 0
        # Postavljanje objekata na nasumične pozicije uz nasumične rotacije oko z-osi
        while not valjana_pozicija and pokusaji < 100:
            x = random.uniform(x_min, x_max)  
            y = random.uniform(y_min, y_max)
            z_rotacija = random.uniform(0, 360)
            pose = robomath.transl(x, y, z) * robomath.rotz(z_rotacija)
            # Provjera kolizije
            detektirana_kolizija = False
            for p_obj in postavljeni_objekti:
                p_x, p_y, _ = globalne_koordinate[p_obj]
                udaljenost = ((x - p_x) ** 2 + (y - p_y) ** 2) ** 0.5
                if udaljenost < min_udaljenost:
                    detektirana_kolizija = True
                    break

            if not detektirana_kolizija:
                valjana_pozicija = True
                postavljeni_objekti.append(f"{i}dio")
                globalne_koordinate[f"{i}dio"] = [x, y, z]
                obj.setPose(pose)    
            else:
                pokusaji += 1

    # Dohvaćanje slike
    kamera = RDK.Item("Camera 1", robolink.ITEM_TYPE_CAMERA)
    spremanje_slike = os.path.join(direktorij, f"slika_{run}.png")  
    RDK.Cam2D_Snapshot(spremanje_slike, kamera)  

    # Učitavanje spremljene slike
    img = Image.open(spremanje_slike)  
    draw = ImageDraw.Draw(img)

    rect_width_mm = 146 
    rect_height_mm = 146
    objects = []  
    
    for ime_objekta, coords in globalne_koordinate.items():
        # Računanje koordinata bbox-a
        x, y, z = coords
        relativne_koordinate = [
            x - pozicija_kamere[0],
            y - pozicija_kamere[1],
            z - pozicija_kamere[2]
        ]

        koordinatni_kamere = [
            relativne_koordinate[0] * orijentacija_kamere[0][0] + relativne_koordinate[1] * orijentacija_kamere[0][1] + relativne_koordinate[2] * orijentacija_kamere[0][2],
            relativne_koordinate[0] * orijentacija_kamere[1][0] + relativne_koordinate[1] * orijentacija_kamere[1][1] + relativne_koordinate[2] * orijentacija_kamere[1][2],
            relativne_koordinate[0] * orijentacija_kamere[2][0] + relativne_koordinate[1] * orijentacija_kamere[2][1] + relativne_koordinate[2] * orijentacija_kamere[2][2]
        ]

        X, Y, Z = koordinatni_kamere
        rect_width_pixels = int((focal_length_pixels * rect_width_mm) / Z)
        rect_height_pixels = int((focal_length_pixels * rect_height_mm) / Z)
        u = (focal_length_pixels * X / Z) + (sensor_width_pixels / 2)
        v = (sensor_height_pixels / 2) + (focal_length_pixels * Y / Z)
        # Pohranjivanje informacija o bbox-u
        bbox = [u - rect_width_pixels / 2, v - rect_height_pixels / 2, rect_width_pixels, rect_height_pixels]
        objects.append({
            "id": ID_counter,
            "area": rect_width_pixels * rect_height_pixels,
            "bbox": bbox,
            "category": int(ime_objekta.replace("dio", "")) - 1
        })

        # Crtanje bbox-a
        gore_lijevo = (bbox[0], bbox[1])
        dolje_desno = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        draw.rectangle([gore_lijevo, dolje_desno], outline="white", width=3)
        pozicija_teksta = (bbox[0], dolje_desno[1] + 5)
        draw.text(pozicija_teksta, ime_objekta, fill="white", font=font)

        ID_counter += 1

    anotirana_slika = os.path.join(direktorij, f"anotirana_slika_{run}.png")
    img.save(anotirana_slika)

    data["image_id"].append(run)
    data["image"].append(spremanje_slike)
    data["width"].append(sensor_width_pixels)
    data["height"].append(sensor_height_pixels)
    data["objects"].append(objects)
# Definiranje strukture Hugging Face skupa podataka
features = Features({
    "image_id": Value(dtype='int64', id=None),
    "image": HFImage(),                                
    "width": Value(dtype='int64', id=None),
    "height": Value(dtype='int64', id=None),            
    "objects": Sequence(                                
        feature={
            "id": Value(dtype='int64', id=None),
            "area": Value(dtype='int64', id=None),
            "bbox": Sequence(feature=Value(dtype='int64', id=None), length=4, id=None),
            "category": Value(dtype='int64', id=None),
        }
    ),
})
# Oblikovanje u Hugging Face skup podatak
hf_dataset = DatasetDict({
    "train": Dataset.from_dict(data, features=features),  
})
# Dijeljenje na Hugging Face repozitorij
hf_dataset.push_to_hub("LovrOP/crna_podloga_250mmm", token=HF_TOKEN)

print("Dataset uploadan!")
