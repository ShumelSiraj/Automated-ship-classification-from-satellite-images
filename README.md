## Automated Ship Classification From Satellite Images

### Dataset
You can download the dataset from [this link](https://www.kaggle.com/datasets/gasgallo/masati-shipdetection). Please refer to the Kaggle dataset download guide for detailed instructions.

### Project Structure
After downloading the dataset, clone this GitHub repository. Make sure you have the following folders in the repository:

1. Code - Contains Python scripts
2. Excel - Includes dataset description and class distribution
3. Final-Group-Presentation - Contains the PowerPoint presentation
4. Final-Group-Project-Report - Includes the PDF report
5. Group Proposal - PDF document
6. Individual-Final-Project-Report - PDF report for individual contributions

Create a "Data" folder in the root directory and place the downloaded dataset inside it. Ensure that the structure looks like this:
Data
- coast
- coast-ship
- detail
- land
- multi
- ship
- water


### Running the Script
Once your project structure is set up, navigate to the "Code" folder and use the following command to run the script:

```bash
cd Code/
python3 train.py --model {model-name}

Replace {model-name} with one of the following options: "VGG16", "VGG19", "Inception", "Resnet", "Xception", or "CNNmodel" based on the model you want to use.

Required Packages
Make sure your environment has the following packages installed:

- tensorflow == 2.11.0
- keras == 2.11.0
- sklearn == 1.2.0
- cv2 == 4.7.0
- numpy == 1.24.1
