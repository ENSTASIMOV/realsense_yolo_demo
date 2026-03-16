# RealSense YOLO Workshop Demo

Démo en temps réel de détection d'objets avec une **Intel RealSense D435i** et **YOLO**, avec affichage en plein écran, estimation de la distance par profondeur et labels affichés en français.

Le projet a été conçu pour un **atelier pédagogique** afin de montrer simplement comment un robot peut :

- voir une scène ;
- reconnaître des objets ou des personnes ;
- estimer leur distance ;
- afficher le résultat en direct sur un écran.

## Aperçu

Le programme récupère les flux couleur et profondeur de la RealSense, applique un modèle YOLO pré-entraîné sur l'image RGB, puis affiche :

- une boîte englobante autour de chaque objet détecté ;
- son nom en français ;
- son score de confiance ;
- sa distance estimée en mètres.

## Fonctionnalités

- Détection d'objets en temps réel avec **Ultralytics YOLO**
- Utilisation des flux **RGB + Depth** de la **RealSense D435i**
- Affichage des **distances** grâce à la profondeur
- **Plein écran** pour une utilisation sur grand écran ou vidéoprojecteur
- **Labels en français**
- Option pour filtrer uniquement certains objets utiles en atelier
- Affichage du **FPS**
- Touche de bascule plein écran / fenêtre

## Matériel utilisé

- Intel RealSense D435i
- PC Linux
- Écran externe optionnel

## Logiciels utilisés

- Ubuntu 22.04
- Python 3.10
- `pyrealsense2`
- `opencv-python`
- `numpy`
- `tkinter`
- `ultralytics`

## Installation

### 1. Installer les dépendances système

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 usbutils v4l-utils
```

### 2. Créer un environnement virtuel

```bash
mkdir -p ~/realsense_yolo_demo
cd ~/realsense_yolo_demo

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Installer les dépendances Python

```bash
pip install pyrealsense2 ultralytics opencv-python numpy
```

## Lancement

```bash
unset LD_LIBRARY_PATH
source ~/realsense_yolo_demo/.venv/bin/activate
cd ~/realsense_yolo_demo
python demo_realsense_yolo.py
```

Au démarrage, le script charge actuellement le modèle `yolo26n.pt`.

## Commandes clavier

- `f` : bascule plein écran / mode fenêtre
- `q` : quitter
- `ESC` : quitter

## Fonctionnement

Le pipeline du projet est le suivant :

1. La **RealSense D435i** fournit :
   - une image couleur ;
   - une carte de profondeur.
2. Le modèle **YOLO** analyse l'image RGB et détecte les objets connus.
3. Pour chaque boîte détectée :
   - on calcule son centre ;
   - on lit la profondeur au centre de la boîte ;
   - on affiche le nom de l'objet, la confiance et la distance.
4. L'image finale est redimensionnée proprement pour remplir tout l'écran.

## Exemple d'affichage

Le programme affiche par exemple :

- `personne 0.91 | 0.54 m`
- `telephone 0.82 | 0.33 m`
- `chaise 0.76 | 0.88 m`

## Objets reconnus

Le modèle utilisé est entraîné sur les **80 classes COCO**.

## Mode atelier

Pour limiter le bruit visuel pendant une démo, il est possible d'activer le filtre suivant :

```python
INTEREST_ONLY = True
```

Les classes conservées par défaut sont :

- person
- apple
- banana
- orange
- bottle
- cup
- cell phone
- book
- mouse
- keyboard
- scissors
- teddy bear

Cela permet de ne garder que les objets les plus parlants pendant un atelier.

## Cas d'usage pédagogique

Ce projet est utile pour illustrer simplement plusieurs notions de robotique et de vision par ordinateur :

- perception RGB-D
- détection d'objets
- estimation de distance
- inférence d'informations à partir d'une caméra
- distinction entre capteur, traitement logiciel et IA

Message clé à faire passer pendant la démo :

> La caméra mesure.
> Le modèle reconnaît.
> Le système estime la distance.
> Le robot peut ensuite agir.

## Auteurs

Projet réalisé par **Jocelyn Deleuil** dans un objectif de démonstration pédagogique autour de la vision robotique.
