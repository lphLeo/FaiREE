This folder contains code for the FairBayes algorithm to estimate the Fair Bayes-optimal classifier under Group Fairness:

Zeng, X., Dobriban, E., & Cheng, G. (2022). Bayes-Optimal Classifiers under Group Fairness. arXiv preprint arXiv:2202.09724.


Run the following command to install the required packages:
pip install -r requirements.txt


### Supported benchmark datasets 
We consider three benchmark datasets for real data analysis. We pre-process the data in the same way as done in the following paper: 
Jaewoong Cho, Gyeongjo Hwang, and Changho Suh. A fair classifier using kernel density
estimation. In Advances in Neural Information Processing Systems, volume 33, pages 15088–
15099, 2020.

- Adult Census Income
D. Dua and C. Graff. UCI machine learning repository, 2017.

- COMPAS Recidivism
J. Angwin, J. Larson, S. Mattu, and L. Kirchner. Machine bias: There’s software
used across the country to 272 predict future criminals. And it’s biased against blacks.
https://www.propublica.org/article/machine-bias-risk-assessments-incriminal-sentencing, 2015.

- Law School Admissions
L. F. Wightman. LSAC national longitudinal bar passage study, 1998.

