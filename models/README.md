# Model Files

This directory should contain the trained model files:

- `best_model_exp_1.pth` - Small model (Experiment 1)
- `best_model_exp_2.pth` - Medium model (Experiment 2) 
- `best_model_exp_3.pth` - Large model (Experiment 3)

## Note for Deployment

If you're deploying this project, you need to:

1. **Train the models** using `my-model-final-collab.ipynb` in Google Colab
2. **Download the trained models** (.pth files) from Colab
3. **Place them in this directory** with the exact names above

## Model Training

To train the models:

1. Open `my-model-final-collab.ipynb` in Google Colab
2. Run all cells to train 3 different model configurations
3. Download the generated `.pth` files
4. Place them in this `models/` directory

The models will be automatically saved during training with the correct names.
