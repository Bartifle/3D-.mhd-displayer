# 3D-mhd-displayer
Displays in point cloud, the data stored in .raw with .mhd file.
Using Plotly (in Browser), Matplotlib (Slices view and 3D view)

## Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/Bartifle/3D-mhd-displayer.git
cd 3D-mhd-displayer
```
### 2. Use a virtual environment (recommended)
#### Create the environment
```bash
python -m venv venv
```
#### Activate the environment
On Linux :
```bash
source venv/bin/activate
```
On Windows
```bash
.\venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
# You can run without arguments (menu mode)
python geant4_3d_viewer.py
```

### Examples of usage
Positional arguments
```bash
# Auto-Deduction of the .raw file
python geant4_3d_viewer.py data.mhd

# Specified .raw file
python geant4_3d_viewer.py data.mhd data.raw
```
Named arguments
```bash
python geant4_3d_viewer.py --mhd data.mhd --raw data.raw
python geant4_3d_viewer.py -m data.mhd -r data.raw
```
Direct visualization
```bash
# Run with Plotly
python geant4_3d_viewer.py data.mhd --mode plotly

# Show all visualizations
python geant4_3d_viewer.py data.mhd --mode all

# Other modes: slices, matplotlib, menu
```
Advanced
```bash
# Change treshold
python geant4_3d_viewer.py data.mhd --threshold 90

# Show all voxels
python geant4_3d_viewer.py data.mhd --show-all

# Combinations
python geant4_3d_viewer.py data.mhd --mode plotly --show-all

# help
python geant4_3d_viewer.py --help
python geant4_3d_viewer.py -h
```
