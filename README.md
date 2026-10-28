# messy-data-eda
Practice real-world data cleaning on a genuinely messy dataset, then use k-means and PCA to explore it.

## Source
**University of Turku, collected 2017–2019 (Zenodo record "Information skills of Finnish basic education teachers")**. Kaggle copy: thedevastator/finnish-basic-education-teacher-ict-skills-and-u.

## Three messy-data problems
- **Hidden missing values**. df.isnull().sum() reports zero missing values, but that's misleading. Three columns are stored as text because blanks are encoded as a single space " ", not as real NaN. Roughly 46% of rows are blank for Self_efficacy and for Inservice_training.
- **The missingness has a pattern**. The two columns are blank together for most of those rows, and those teachers differ from the rest (e.g. older, lower information-skills scores). That suggests these questions weren't answered or asked for a whole subgroup (the cause is likely structural, such as a different questionnaire version or round, but verify in the source documentation). It is not random noise, so filling blanks with the median would be inventing data for nearly half the sample.
- **29 exact duplicate rows**, with no ID column to tell whether they're double submissions or coincidentally identical teachers.

## Principal Component Analysis (PCA)

# 1. Standardise and conter the data
- Subtract the mean of each feature so the data is centered at zero, and scale features (often by dividing by the standard deviation) so variable with larger units do not dominate.

z = lambda s: (s - s.mean()) / s.std()

represents...

$$
Z=\frac{X-µ}{σ}
$$

where...
- "µ" is the mean of independent features
- "σ" is the standard deviation of independent features



In this research, the standardised data of "Self_efficacy" is substracted by the standardised data of "Information_skills", which reveals the "Conficence_gap" to suggests if teachers over- or underestimate their digital skills.
df['Confidence_gap'] = z(df['Self-efficacy]) - z(df[Information_skills])

# 2. Compute the covariance matrix
