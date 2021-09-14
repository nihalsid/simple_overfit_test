# Overfitting Coordinate MLP to Colors

The objective of this repo is to simply overfit an MLP such that it takes coordinates as input and outputs colors. The data to overfit is provided in `dataset/data`. This data originally comes from a mesh with vertex colors.

Until now, I've been unsuccessful in achieving this objective.

### File Structure


| Folder                    | Description                            |
|---------------------------|----------------------------------------|
| `dataset/data`    | Contains the mesh data the needs to be overfit      |
| `dataset/mesh_data.py` | Contains dataset class for training    |
| `model/net_texture.py`         | MLP with positional encoding|
| `trainer/train_overfit.py` | Training module                        |
| runs                   | Experiment checkpoints and outputs |

### Requirements

Requirements are provided in `requirements.txt` file. To install them, run

`
pip install -r requirements.txt 
`

### Training 

To train the MLP simply run

`
python trainer/train_overfit.py
`
