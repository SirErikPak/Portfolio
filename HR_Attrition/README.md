# hr_attrition_analysis
DSC424 Milestone 5 Final Project: HR Attrition Data
Author: Erik Pak, Team (Data Crew)

Milestone 3 Review:
Initial Hypothesis:
From the milestone three visual plots, we inferred that males who work more overtime and are single have more attrition and are most likely in a lower income bracket. We need to analyze this more in our final project. We also see a significant attrition rate among young employees, a sector requiring additional analysis.
Aspect line of Analysis: Regression
 
Dataset Description: [NOTE: Initial Explanatory Graphs is submitted in milestone 3]
As mentioned in the previous milestone, we cleaned and preprocessed data, converted variables to factors, and created dummies. The variables involved in the analysis are as follows. There are a couple of categorical variables, which will be explained in the final report.

Our response variable is attrition; with this approach, we want to measure the performance of different models to predict attrition. We conducted regression to analyze our data using the technique below.
 
• Logistic Regression, Regularized subset, Ridge Regression, Lasso Regression, Elasticnet Regression.
 
SSince our class had data imbalance, we oversampled the data and verified the distribution was the same. (Detail graphs in the final report)

Bar Graph showing Before and after Oversampling
To prepare the data set, we converted our variables to dummies and applied the “Principal component analysis Technique” to reduce dimensionality and determine the number of factors. We then performed a “Principal Factor analysis” and determined the factors used for regression. Our analysis suggested 16 factors, but we took 13 factors, covering 58% of the variance. (Details in the final report)
 
Logistic Regression
We performed a logistic regression dividing data into training and testing with manual variable selection and measured the performance. The model turned out good with selected variable being significant at p < 0.05 having a goodness of fit of chi square = 16.73 , p < 2.2e-16 ***.
log(p/(1-p)) = -0.08 - 0.42(RC6) -0.30(RC1) + 0.21(RC3) + 0.12(RC2) - 0.44(RC4) - 0.25(RC9)
               ​​​+ 0.35(RC8) - 0.11(RC13) - 0.13(RC12)
RegSubset Regression
We performed a Reg subset regression using “adjR2” and “bic” scale to measure the performance and found out the model with adjR2 did better after removing the non-significant variables and the performance was same as logistic.
 
   
(adjR2 Equation)    log(p/(1-p)) = -0.08 - 0.42(RC6) -0.30(RC1) + 0.21(RC3) + 0.12(RC2) - 0.44(RC4) - 0.25(RC9)
              ​​​​ + 0.35(RC8) - 0.11(RC13) - 0.13(RC12)
Ridge Regression
We then performed a ridge regression as another technique to verifed the performance, we select lambda.min for our lambda parameter as we want to have maximum confidence interval to capture most of the true positives.

Lasso Regression
We then performed a lasso regression to verify the performance again and it performed the almost similar.
 
  
Elastic net Regression
For elasticnet we ran a for loop with a range of alpha 0 to 1 with an increment of 0.01 and captured the performance which did better, and it reported that model at alpha =0.55 did better with performance which we see is same as the performance of lasso.
 
Performance Comparison of Models
From the Performance matrix we see Lasso is doing slightly better than all other model which is also reported in elastic net.
 
<img width="309" alt="image" src="https://github.com/jitendra3010/hr_attrition_analysis/assets/53829596/6d916079-8503-4463-8a5e-f200bbc4a637">

 
Conclusion: If we look at the model equation, we have some of the highest coefficients for RC8, which is a factor of Overtime With research scientists; RC3, which is a factor of Sales and marketing employees; RC2, which is a factor of HR Profile, and RC6, which affects negatively, which is a factor of Manager Profile. These all factors have a significant influence on attrition, which is kind of supporting our initial hypothesis.
[NOTE: More details in the Final Report]
