# Predicting-Student-Retention-in-4-Year-University
The purpose of this project is to create a logistic regression model that may be used to forecast the student retention rate at a four-year institution. Data on student demographics, socioeconomic status, and first-semester academic achievement are used to construct the model. The goal of this approach is to assist university management in spotting potentially at-risk students so that effective preventative measures may be taken.

##Table of Contents
-Data (#data)
-Exploratory Data Analysis (#EDA)
-Data Preprocessing (#data-preprocessing)
-Model Building (#model-building)
-Evaluation (#evaluation)
-Conclusion (#conclusion)

##Data
This dataset provides a comprehensive view of students enrolled in undergraduate higher education.  It includes demographic data, social-economic factors and academic performance information that can be used to analyze the possible predictors of student dropout and academic success. This dataset contains multiple disjoint databases consisting of relevant information available at the time of enrollment, such as marital status, whether or not they are a scholarship holder and more. Additionally, this data can be used to estimate overall student performance at the end of each semester by assessing approved curricular units as well as their academic success. Finally, we have unemployment rate, inflation rate and GDP from the region which can help us further understand how economic factors play into student dropout rates or academic success outcomes. This powerful analysis tool will provide valuable insight into what motivates students to stay in school or abandon their studies.

##Exploratory Data Analysis
I first did some exploratory data analysis to see if I could see any trends or patterns that would have an impact on the retention rate before I built the predictive model. To better grasp the distribution of the data and the relationships between the variables, I created many charts and graphs. I also ran statistical analyses to see whether there were any substantial changes in retention rates based on demographic variables including gender, socioeconomic status, and academic standing.

##Data Preprocessing
After conducting the exploratory data analysis, I preprocessed the data to prepare it for the logistic regression model. This involved handling missing values and encoding categorical variables. I also split the data into training and testing sets in a 70:30 ratio.

##Model Building
I used the training data to create a logistic regression model. Accuracy, precision, and recall were measured by first training the model on a portion of the data and then testing it on the full dataset. To further evaluate the efficacy of the model, we displayed the ROC curve and determined the area under the curve (AUC).

##Evaluation
The logistic regression model achieved an accuracy of 0.90 and an AUC of 0.92 on the testing set, indicating that it is a good predictor of the student retention rate. We also analyzed the coefficients of the model to identify the factors that are most strongly associated with the retention rate. These factors include whether their tuition fees was up to date, whether or not they were a scholarship holder, and if they were a debtor or not.

##Conclusion
In conclusion, this project demonstrates the effectiveness of a logistic regression model in predicting the student retention rate of a 4-year university. The model can be used to identify at-risk students and provide early interventions to improve their retention rates. Future work includes improving the model's accuracy by incorporating additional data sources and exploring other machine learning algorithms and potentially deploying the model into a 4 year university.
