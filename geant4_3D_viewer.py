import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
import plotly.graph_objects as go
import plotly.express as px
import argparse
import os
import sys

class Geant4Viewer:
    def __init__(self, mhd_file, raw_file=None):
        """
        Initialise le visualiseur avec les fichiers .mhd et .raw
        Si raw_file n'est pas spécifié, on essaie de le déduire du fichier .mhd
        """
        self.mhd_file = mhd_file
        
        # Si pas de fichier raw spécifié, on essaie de le déduire
        if raw_file is None:
            # Remplacer .mhd par .raw
            if mhd_file.endswith('.mhd'):
                self.raw_file = mhd_file.replace('.mhd', '.raw')
            else:
                raise ValueError("Fichier .mhd requis ou spécifiez le fichier .raw")
        else:
            self.raw_file = raw_file
            
        # Vérifier que les fichiers existent
        if not os.path.exists(self.mhd_file):
            raise FileNotFoundError(f"Fichier .mhd introuvable: {self.mhd_file}")
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(f"Fichier .raw introuvable: {self.raw_file}")
            
        self.data = None
        self.metadata = {}
        
        print(f"Chargement des fichiers:")
        print(f"  .mhd: {self.mhd_file}")
        print(f"  .raw: {self.raw_file}")
        
        self.load_data()
    
    def parse_mhd_file(self):
        """
        Parse le fichier .mhd pour extraire les métadonnées
        """
        with open(self.mhd_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split(' = ', 1)
                    self.metadata[key] = value
        
        # Convertir les valeurs importantes
        self.dim_size = [int(x) for x in self.metadata['DimSize'].split()]
        self.element_spacing = [float(x) for x in self.metadata['ElementSpacing'].split()]
        self.offset = [float(x) for x in self.metadata['Offset'].split()]
        
        print(f"Dimensions: {self.dim_size}")
        print(f"Espacement: {self.element_spacing} mm")
        print(f"Offset: {self.offset} mm")
    
    def load_raw_data(self):
        """
        Charge les données binaires du fichier .raw
        """
        # Geant4 utilise généralement des doubles (8 bytes)
        element_type = self.metadata.get('ElementType', 'MET_DOUBLE')
        
        if element_type == 'MET_DOUBLE':
            dtype = np.float64
            bytes_per_element = 8
        elif element_type == 'MET_FLOAT':
            dtype = np.float32
            bytes_per_element = 4
        else:
            dtype = np.float64
            bytes_per_element = 8
        
        total_elements = np.prod(self.dim_size)
        
        with open(self.raw_file, 'rb') as f:
            data_bytes = f.read()
        
        # Convertir en array numpy
        data_array = np.frombuffer(data_bytes, dtype=dtype)
        
        # Reshape selon les dimensions
        self.data = data_array.reshape(self.dim_size)
        
        print(f"Données chargées: {self.data.shape}")
        print(f"Énergie totale déposée: {np.sum(self.data):.4e} MeV")
        print(f"Énergie max: {np.max(self.data):.4e} MeV")
        print(f"Nombre de voxels non-zéro: {np.count_nonzero(self.data)}")
    
    def load_data(self):
        """
        Charge les métadonnées et les données
        """
        self.parse_mhd_file()
        self.load_raw_data()
    
    def plot_slice_viewer(self):
        """
        Visualiseur de coupes avec slider interactif
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Visualiseur de coupes - Dépôt d\'énergie Geant4')
        
        # Position initiale au centre
        z_center = self.dim_size[2] // 2
        y_center = self.dim_size[1] // 2
        x_center = self.dim_size[0] // 2
        
        # Coupes initiales
        slice_xy = self.data[:, :, z_center]
        slice_xz = self.data[:, y_center, :]
        slice_yz = self.data[x_center, :, :]
        
        # Affichage
        im1 = axes[0].imshow(slice_xy.T, origin='lower', cmap='viridis')
        axes[0].set_title(f'Coupe XY (z={z_center})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        im2 = axes[1].imshow(slice_xz.T, origin='lower', cmap='viridis')
        axes[1].set_title(f'Coupe XZ (y={y_center})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        im3 = axes[2].imshow(slice_yz.T, origin='lower', cmap='viridis')
        axes[2].set_title(f'Coupe YZ (x={x_center})')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        
        # Colorbars
        plt.colorbar(im1, ax=axes[0], label='Énergie (MeV)')
        plt.colorbar(im2, ax=axes[1], label='Énergie (MeV)')
        plt.colorbar(im3, ax=axes[2], label='Énergie (MeV)')
        
        plt.tight_layout()
        plt.show(block=False)  # Non-bloquant
        print("Ferme la fenêtre des coupes pour continuer vers la visualisation 3D...")
    
    def plot_3d_matplotlib(self, threshold_percentile=90, show_all=False):
        """
        Visualisation 3D avec matplotlib (voxels)
        """
        if show_all:
            # Montrer tous les voxels avec énergie > 0
            mask = self.data > 0
        else:
            # Appliquer un seuil pour ne montrer que les voxels avec suffisamment d'énergie
            threshold = np.percentile(self.data[self.data > 0], threshold_percentile)
            mask = self.data > threshold
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Créer les coordonnées des voxels
        x, y, z = np.meshgrid(
            np.arange(self.dim_size[0]) * self.element_spacing[0] + self.offset[0],
            np.arange(self.dim_size[1]) * self.element_spacing[1] + self.offset[1],
            np.arange(self.dim_size[2]) * self.element_spacing[2] + self.offset[2],
            indexing='ij'
        )
        
        # Filtrer selon le masque
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        values_filtered = self.data[mask]
        
        # Scatter plot 3D avec couleurs selon l'énergie
        scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=values_filtered, cmap='plasma', s=20, alpha=0.6)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Dépôt d\'énergie 3D {"(tous voxels)" if show_all else f"(>{threshold_percentile}e percentile)"}')
        
        plt.colorbar(scatter, label='Énergie déposée (MeV)')
        plt.show(block=False)  # Non-bloquant
        print("Ferme la fenêtre 3D matplotlib pour continuer...")
    
    def plot_3d_plotly(self, threshold_percentile=85, show_all=False):
        """
        Visualisation 3D interactive avec Plotly
        """
        if show_all:
            # Montrer tous les voxels avec énergie > 0
            mask = self.data > 0
        else:
            # Appliquer un seuil
            threshold = np.percentile(self.data[self.data > 0], threshold_percentile)
            mask = self.data > threshold
        
        # Créer les coordonnées
        x, y, z = np.meshgrid(
            np.arange(self.dim_size[0]) * self.element_spacing[0] + self.offset[0],
            np.arange(self.dim_size[1]) * self.element_spacing[1] + self.offset[1],
            np.arange(self.dim_size[2]) * self.element_spacing[2] + self.offset[2],
            indexing='ij'
        )
        
        # Filtrer
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        values_filtered = self.data[mask]
        
        # Créer le scatter 3D
        fig = go.Figure(data=go.Scatter3d(
            x=x_filtered,
            y=y_filtered,
            z=z_filtered,
            mode='markers',
            marker=dict(
                size=3,
                color=values_filtered,
                colorscale='Plasma',
                colorbar=dict(title="Énergie (MeV)"),
                opacity=0.8
            ),
            text=[f'E: {v:.2e} MeV' for v in values_filtered],
            hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Visualisation 3D interactive - Dépôt d\'énergie Geant4 {"(tous voxels)" if show_all else ""}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    

    
    def get_statistics(self):
        """
        Affiche les statistiques des données
        """
        print("\n=== STATISTIQUES ===")
        print(f"Dimensions du volume: {self.dim_size}")
        print(f"Résolution spatiale: {self.element_spacing} mm")
        print(f"Volume total: {np.prod(self.dim_size) * np.prod(self.element_spacing):.1f} mm³")
        print(f"Nombre total de voxels: {np.prod(self.dim_size)}")
        print(f"Voxels avec énergie > 0: {np.count_nonzero(self.data)}")
        print(f"Énergie totale déposée: {np.sum(self.data):.4e} MeV")
        print(f"Énergie moyenne (voxels non-zéro): {np.mean(self.data[self.data > 0]):.4e} MeV")
        print(f"Énergie maximale: {np.max(self.data):.4e} MeV")
        print(f"Énergie minimale (>0): {np.min(self.data[self.data > 0]):.4e} MeV")

def parse_arguments():
    """
    Parse les arguments de ligne de commande
    """
    parser = argparse.ArgumentParser(
        description='Visualiseur 3D pour les données de dépôt d\'énergie Geant4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Exemples d'utilisation:
            python geant4_3d_viewer.py data.mhd
            python geant4_3d_viewer.py data.mhd data.raw
            python geant4_3d_viewer.py --mhd data.mhd --raw data.raw
            python geant4_3d_viewer.py -m data.mhd -r data.raw --mode plotly
        '''
    )
    
    # Arguments positionnels
    parser.add_argument('mhd_file', nargs='?', help='Fichier .mhd (métadonnées)')
    parser.add_argument('raw_file', nargs='?', help='Fichier .raw (données binaires, optionnel)')
    
    # Arguments optionnels
    parser.add_argument('-m', '--mhd', dest='mhd_file_alt', help='Fichier .mhd (alternative)')
    parser.add_argument('-r', '--raw', dest='raw_file_alt', help='Fichier .raw (alternative)')
    
    # Mode de visualisation
    parser.add_argument('--mode', choices=['menu', 'slices', 'matplotlib', 'plotly', 'all'], default='menu', help='Mode de visualisation (défaut: menu)')
    
    # Options d'affichage
    parser.add_argument('--show-all', action='store_true', help='Montrer tous les voxels (pas de seuil)')
    parser.add_argument('--threshold', type=float, default=50, help='Percentile de seuil pour filtrer les voxels (défaut: 50)')
    
    args = parser.parse_args()
    
    # Déterminer les fichiers à utiliser
    mhd_file = args.mhd_file or args.mhd_file_alt
    raw_file = args.raw_file or args.raw_file_alt
    
    if not mhd_file:
        parser.error("Un fichier .mhd est requis")
    
    return mhd_file, raw_file, args

def interactive_file_selection():
    """
    Sélection interactive des fichiers si pas d'arguments
    """
    print("=== SÉLECTION DES FICHIERS ===")
    
    while True:
        mhd_file = input("Chemin vers le fichier .mhd: ").strip()
        if os.path.exists(mhd_file):
            break
        print(f"Fichier introuvable: {mhd_file}")
    
    # Essayer de déduire le fichier .raw
    raw_file_auto = mhd_file.replace('.mhd', '.raw') if mhd_file.endswith('.mhd') else None
    
    if raw_file_auto and os.path.exists(raw_file_auto):
        use_auto = input(f"Utiliser {raw_file_auto} ? (o/n): ").strip().lower()
        if use_auto in ['o', 'oui', 'y', 'yes', '']:
            return mhd_file, raw_file_auto
    
    while True:
        raw_file = input("Chemin vers le fichier .raw: ").strip()
        if os.path.exists(raw_file):
            break
        print(f"Fichier introuvable: {raw_file}")
    
    return mhd_file, raw_file

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        # Essayer de parser les arguments
        if len(sys.argv) > 1:
            mhd_file, raw_file, args = parse_arguments()
        else:
            # Sélection interactive si pas d'arguments
            mhd_file, raw_file = interactive_file_selection()
            # Créer un objet args par défaut
            class DefaultArgs:
                mode = 'menu'
                show_all = False
                threshold = 5
            args = DefaultArgs()
        
        # Créer le visualiseur
        viewer = Geant4Viewer(mhd_file, raw_file)
        
        # Afficher les statistiques
        viewer.get_statistics()
        
        # Exécuter selon le mode choisi
        if args.mode == 'menu':
            # Menu interactif
            print("\n=== MENU DE VISUALISATION ===")
            print("1. Visualiseur de coupes")
            print("2. Visualisation 3D (matplotlib)")
            print("3. Visualisation 3D interactive (Plotly)")
            print("4. Toutes les visualisations")
            
            choice = input("\nChoisis une option (1-4): ")
            
            if choice == "1":
                print("\nVisualiseur de coupes...")
                viewer.plot_slice_viewer()
            elif choice == "2":
                print("\nVisualisation 3D (matplotlib)...")
                viewer.plot_3d_matplotlib(show_all=args.show_all, threshold_percentile=args.threshold)
            elif choice == "3":
                print("\nVisualisation 3D interactive (Plotly)...")
                viewer.plot_3d_plotly(show_all=args.show_all, threshold_percentile=args.threshold)
            elif choice == "4":
                print("\n1. Visualiseur de coupes...")
                viewer.plot_slice_viewer()
                
                input("Appuie sur Entrée pour continuer vers la visualisation 3D matplotlib...")
                print("\n2. Visualisation 3D (matplotlib)...")
                viewer.plot_3d_matplotlib(show_all=args.show_all, threshold_percentile=args.threshold)
                
                input("Appuie sur Entrée pour continuer vers la visualisation 3D interactive...")
                print("\n3. Visualisation 3D interactive (Plotly)...")
                viewer.plot_3d_plotly(show_all=args.show_all, threshold_percentile=args.threshold)
            else:
                print("Option invalide!")
                
        elif args.mode == 'slices':
            viewer.plot_slice_viewer()
        elif args.mode == 'matplotlib':
            viewer.plot_3d_matplotlib(show_all=args.show_all, threshold_percentile=args.threshold)
        elif args.mode == 'plotly':
            viewer.plot_3d_plotly(show_all=args.show_all, threshold_percentile=args.threshold)
        elif args.mode == 'all':
            viewer.plot_slice_viewer()
            input("Appuie sur Entrée pour continuer...")
            viewer.plot_3d_matplotlib(show_all=args.show_all, threshold_percentile=args.threshold)
            input("Appuie sur Entrée pour continuer...")
            viewer.plot_3d_plotly(show_all=args.show_all, threshold_percentile=args.threshold)
        
        print("\nTerminé!")
        
    except KeyboardInterrupt:
        print("\n\nArrêt demandé par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErreur: {e}")
        sys.exit(1)