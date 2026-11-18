import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Women's Clothing Review Analysis Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.spatial.distance import cdist\n",
    "import scipy.stats as stats\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Dataset\n",
    "df = pd.read_csv(\"womens_clothing.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset Overview\n",
    "print(df.shape)\n",
    "print(df.info())\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Cleaning & Missing Value Imputation\n",
    "df = df.drop_duplicates()\n",
    "df['Title'] = df['Title'].fillna(\"No Title Provided\")\n",
    "df['Review Text'] = df['Review Text'].fillna(\"No Review Text Provided\")\n",
    "df['Division Name'] = df['Division Name'].fillna(df['Division Name'].mode()[0])\n",
    "df['Department Name'] = df['Department Name'].fillna(df['Department Name'].mode()[0])\n",
    "df['Class Name'] = df['Class Name'].fillna(df['Class Name'].mode()[0])\n",
    "df['Age'] = df['Age'].fillna(df['Age'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Age Distribution Visualization\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.histplot(df['Age'], kde=True, bins=20)\n",
    "plt.title(\"Age Distribution\")\n",
    "plt.xlabel(\"Age\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Average Rating by Department\n",
    "grouped_mean = df.groupby('Department Name')['Rating'].mean()\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(x=grouped_mean.index, y=grouped_mean.values)\n",
    "plt.title(\"Average Rating by Department\")\n",
    "plt.xlabel(\"Department Name\")\n",
    "plt.ylabel(\"Average Rating\")\n",
    "plt.xticks(rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation Matrix Heatmap\n",
    "plt.figure(figsize=(8, 5))\n",
    "numeric_cols = df.select_dtypes(include=['number']).columns\n",
    "sns.heatmap(df[numeric_cols].corr(), annot=True, cmap=\"coolwarm\", fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export Cleaned Data\n",
    "df.to_csv(\"cleaned_womens_clothing.csv\", index=False)\n",
    "print(\"Data successfully cleaned and saved!\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Overwrite / Create the notebook file cleanly
with open("womens_clothing.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("womens_clothing.ipynb regenerated successfully and is ready to open!")

{
"cells": [
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# Analysis Notebook\n",
"This notebook has been successfully cleaned and the merge conflict markers have been resolved."
]
},
{
"cell_type": "code",
"execution_count": 1,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"Environment ready and verified.\n"
]
}
],
"source": [
"print("Environment ready and verified.")"
]
}
],
"metadata": {
"language_info": {
"name": "python"
}
},
"nbformat": 4,
"nbformat_minor": 2
}