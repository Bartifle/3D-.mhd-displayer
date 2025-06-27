import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
import plotly.graph_objects as go
import plotly.express as px

class Geant4Viewer:
    def __init__(self, mhd_file, raw_file):
        """
        Initialise le visualiseur avec les fichiers .mhd et .raw
        """
        self.mhd_file = mhd_file
        self.raw_file = raw_file
        self.data = None
        self.metadata = {}
        
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
        scatter = ax.scatter(x_filtered, y_filtered, z_filtered, 
                           c=values_filtered, cmap='plasma', 
                           s=20, alpha=0.6)
        
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
    
    def plot_volume_rendering(self):
        """
        Rendu volumique avec Plotly
        """
        # Sous-échantillonner si nécessaire pour les performances
        step = max(1, max(self.dim_size) // 50)
        data_subsampled = self.data[::step, ::step, ::step]
        
        # Coordonnées
        x = np.arange(0, self.dim_size[0], step) * self.element_spacing[0] + self.offset[0]
        y = np.arange(0, self.dim_size[1], step) * self.element_spacing[1] + self.offset[1]
        z = np.arange(0, self.dim_size[2], step) * self.element_spacing[2] + self.offset[2]
        
        fig = go.Figure(data=go.Volume(
            x=x,
            y=y,
            z=z,
            value=data_subsampled.flatten(),
            isomin=0.1 * np.max(data_subsampled),
            isomax=np.max(data_subsampled),
            opacity=0.1,
            surface_count=15,
            colorscale='Plasma'
        ))
        
        fig.update_layout(
            title='Rendu volumique - Dépôt d\'énergie',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)'
            )
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

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplace par tes fichiers
    mhd_file = "test008-edep_edep.mhd"
    raw_file = "test008-edep_edep.raw"
    
    # Créer le visualiseur
    viewer = Geant4Viewer(mhd_file, raw_file)
    
    # Afficher les statistiques
    viewer.get_statistics()
    
    # Menu interactif
    print("\n=== MENU DE VISUALISATION ===")
    print("1. Visualiseur de coupes")
    print("2. Visualisation 3D (matplotlib)")
    print("3. Visualisation 3D interactive (Plotly)")
    print("4. Rendu volumique")
    print("5. Toutes les visualisations")
    
    choice = input("\nChoisis une option (1-5): ")
    
    if choice == "1":
        print("\nVisualiseur de coupes...")
        viewer.plot_slice_viewer()
    elif choice == "2":
        print("\nVisualisation 3D (matplotlib)...")
        viewer.plot_3d_matplotlib(show_all=True)
    elif choice == "3":
        print("\nVisualisation 3D interactive (Plotly)...")
        viewer.plot_3d_plotly(show_all=True)
    elif choice == "4":
        print("\nRendu volumique...")
        viewer.plot_volume_rendering()
    elif choice == "5":
        print("\n1. Visualiseur de coupes...")
        viewer.plot_slice_viewer()
        
        input("Appuie sur Entrée pour continuer vers la visualisation 3D matplotlib...")
        print("\n2. Visualisation 3D (matplotlib)...")
        viewer.plot_3d_matplotlib(show_all=True)
        
        input("Appuie sur Entrée pour continuer vers la visualisation 3D interactive...")
        print("\n3. Visualisation 3D interactive (Plotly)...")
        viewer.plot_3d_plotly(show_all=True)
        
        input("Appuie sur Entrée pour continuer vers le rendu volumique...")
        print("\n4. Rendu volumique...")
        viewer.plot_volume_rendering()
    else:
        print("Option invalide!")
        
    print("\nTerminé!")