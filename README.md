# ProjectAlpha

| Column Name      | Description                                            | Type        |
| ---------------- | ------------------------------------------------------ | ----------- |
| `age`            | Age of the individual                                  | Numerical   |
| `workclass`      | Type of employment (e.g., Private, Self-emp, etc.)     | Categorical |
| `fnlwgt`         | Final weight (used by the census for population stats) | Numerical   |
| `education`      | Education level (e.g., Bachelors, HS-grad)             | Categorical |
| `education-num`  | Number representing education level                    | Numerical   |
| `marital-status` | Marital status                                         | Categorical |
| `occupation`     | Type of job (e.g., Tech-support, Sales, etc.)          | Categorical |
| `relationship`   | Relationship (e.g., Wife, Not-in-family)               | Categorical |
| `race`           | Race of the individual                                 | Categorical |
| `sex`            | Gender                                                 | Categorical |
| `capital-gain`   | Income from investment sources like stocks             | Numerical   |
| `capital-loss`   | Loss from investment                                   | Numerical   |
| `hours-per-week` | Hours worked per week                                  | Numerical   |
| `native-country` | Country of origin                                      | Categorical |
| `income`         | **Target** — whether income is `>50K` or `<=50K`       | Categorical |



📦 What is fnlwgt?
The fnlwgt (final weight) stands for final sampling weight. It comes from how the U.S. Census Bureau samples and scales individuals to represent the entire U.S. population.

It’s a numerical column used during survey design to indicate how representative each person is in the dataset.

🧮 What does it mean practically?
If fnlwgt = 1000, it means this person represents 1000 similar people in the U.S. population.

So two people might have the same age, job, and income — but different fnlwgt values — because one is more statistically representative than the other based on how the sample was drawn.

📊 Why Does the Census Use It?
The U.S. Census uses stratified sampling: sampling certain groups more or less heavily based on demographics.

fnlwgt is used to rebalance the sample so it better reflects the true population proportions.

🧠 Should You Use fnlwgt in Your ML Model?
✅ Use it only if:
You're doing population-level statistics, e.g., “how many Americans make over $50K.”

You're building a weighted model that mimics the real-world population.

🚫 Usually don't use it if:
You're doing pure predictive modeling (e.g., to predict income for individuals).

You're training models like Random Forest, Logistic Regression, etc. where weights
can introduce unnecessary noise.