# EDM
Educational Data Mining for Anomaly Detection on Student Assessment in E-Learning Environments

## Abstract
According to the legislation of Brazil’s Ministry of Education (MEC),  the student assessment in distance learning programs (e-learning) is based on face-to-face exams at an educational center and online activities. The legislation also requires that the face-to-face exams must have the heaviest weight in the final performance. Given this, the present article seeks to question whether this requirement is generating students who make minimal use of the resources offered by the e-learning platforms but still achieve passing grades because of the face-to-face exams weight, thus affecting the effectiveness of distance learning. For such purpose, a model has been defined and validated using the Isolation Forest algorithm to identify these anomalies, after which, the behavior of the students regarding use of the online platform was analyzed.

## File Structure
- Dataset 
    - baseRaw.csv - Dataset extracted from Moodle platform.
    - DataDictionary.xlsx - Description of all variables in the dataset.
    - Subsets - Original dataset divided by course and semester.

- PreProcessing.py - Formats the dataset and generates subsets for analysis.

- Modeling Notebook - Applies the technique to a subset.
