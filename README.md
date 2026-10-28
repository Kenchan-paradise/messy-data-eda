# messy-data-eda
Practice real-world data cleaning on a genuinely messy dataset, then use k-means and PCA to explore it.

## Source
**University of Turku, collected 2017–2019 (Zenodo record "Information skills of Finnish basic education teachers")**. Kaggle copy: thedevastator/finnish-basic-education-teacher-ict-skills-and-u.

## Three messy-data problems
- **Hidden missing values**. df.isnull().sum() reports zero missing values, but that's misleading. Three columns are stored as text because blanks are encoded as a single space " ", not as real NaN. Roughly 46% of rows are blank for Self_efficacy and for Inservice_training.
- **The missingness has a pattern**. The two columns are blank together for most of those rows, and those teachers differ from the rest (e.g. older, lower information-skills scores). That suggests these questions weren't answered or asked for a whole subgroup (the cause is likely structural, such as a different questionnaire version or round, but verify in the source documentation). It is not random noise, so filling blanks with the median would be inventing data for nearly half the sample.
- **29 exact duplicate rows**, with no ID column to tell whether they're double submissions or coincidentally identical teachers.

# Principal Component Analysis (PCA)


## 1. Standardise and centre the data
- Subtract the mean of each feature so the data is centred at zero, and scale features (often by dividing by the standard deviation) so variable with larger units do not dominate.

`z = lambda s: (s - s.mean()) / s.std()`

represents...

$$
Z=\frac{X-µ}{σ}
$$

where...
- "µ" is the mean of independent features
- "σ" is the standard deviation of independent features

In this research, the standardised data of "**Self_efficacy**" is substracted by the standardised data of "**Information_skills**", which reveals the "**Conficence_gap**" to suggests if teachers over- or underestimate their digital skills.

`df['Confidence_gap'] = z(df['Self-efficacy]) - z(df[Information_skills])`

The result reads as follows:
- **Positive**: The teacher rates themself higher (relative to other teachers) than their test score of information skills suggests.
- **Negative**: They rate themself lower than their test suggests.
- **Near 0**: Self-perseption matches the test.

Q. Why standardise *before* subtracting?<br />
A. **Self_efficacy** runs between 0 and 1 and **Information_skills** runs roughly between 0 and 2. So, sbtracting the raw values would be unfair.

## 2. Compute the covariance matrix
- Calculate a sequre matrix showing how every feature in the dataset relates or varies with every other feature.

## 3. Perform eigen decomposition
- Find the **eigenvalues** and **eigenvectors** of the covariance matrix. The eigenvectors define the new directional axes (**principal components**), and the eigenvalues measure the amount of variance captured by each axis.

## 4. Select feature vectors
- Sort the eigenvectors by their corresponding eigenvalues from highest to lowest, and choose the top k eigenvectors that capture the desired amount of total variance.

## 5. Project the data
- Multiply the original standardized dataset by the matrix of selected eigenvectors to transform the data into the new lower-dimensional subspace.
