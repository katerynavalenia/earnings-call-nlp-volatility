
Applied Big Data Analytics in Finance
Master 2 — Paris 1 Pantheon Sorbonne
Thomas Rigou
Final Evaluation Guidelines
Submission Deadline: March 30, 12PM
# 1. Overview
The final evaluation for this course is a group project carried out in teams of 3 to 4 students. The objective is to formulate a research question about financial text, build a complete analytical pipeline to investigate it, and present the findings during the final class session on March 31, 2026.
The project is designed to assess the ability to apply the techniques covered throughout the course — from data collection and preprocessing to model fine-tuning and result interpretation — within a coherent research framework.
# 2. Task Description
## Each group must select a research question grounded in financial text analysis. The project proceeds through the following stages:
Define a research question.
Choose a qualitative measure that helps answer the question — for example, sentiment analysis, topic classification, tone detection, emotion classification, share of boilerplate language, novelty of analyst questions relative to prepared remarks, or another construct that can be justified.
Generate or obtain a training dataset using an LLM-based labeling approach, a lexicon-based method, looking for a dataset on HuggingFace...
Train a model on your training data to transform textual information into a numerical variable.
Analyze and interpret the results — connect the model’s outputs back to the research question through appropriate statistical or econometric analysis.
# 3. Deliverables and Weighting
## The final grade is composed of three equally weighted components:

# 4. Detailed Evaluation Criteria
# 4.1  Report
## The report should read as a short research paper. The following aspects will receive particular attention:
Coherence of modeling choices. The choice of model architecture, loss function, training data construction, and evaluation metrics must be consistent with the stated research question and the nature of the construct being measured.
Econometric soundness. Regressions and statistical analyses must be methodologically rigorous. Common errors that will be penalized include look-ahead bias in forecasting settings, multicollinearity among regressors, poorly chosen or unjustified fixed effects, and any form of leakage between the dependent variable and the explanatory variables.
Interpretability of results. Findings must be framed in terms that are meaningful and accessible to a human reader. Results should be expressed in natural, interpretable units rather than in standardized units.
Avoid: "A one standard deviation increase in <measure> leads to a 0.2 standard deviation increase in <outcome>."
Prefer: "A 1% increase in the share of discussion devoted to <topic> during the call is associated with 0.3% lower returns over the following week."
# 4.2  Code and Data
## The codebase will be evaluated on structure and readability:
Project architecture. The repository should be well-organized with dedicated folders and scripts for each stage of the pipeline (e.g., data collection, preprocessing, labeling, training, evaluation, analysis). A clear separation of concerns is expected.
Code clarity. Scripts should be clean, readable, and appropriately documented. Overly complex or unnecessarily long scripts will be viewed unfavorably. The examiners reserve the right to ask questions about specific code snippets during the presentation if any part of the codebase appears difficult to understand.
Data. The number of transcripts successfully scraped will be evaluated with the maximum points being awarded for groups that fetched at least 100 000 transcripts.
# 4.3  Presentation
Each group will present its work in a 10- to 15-minute session followed by questions. The presentation should convey a clear narrative arc, from the research question, through the methodology and to the key findings.
An important note on delivery: students are expected to present their work with sufficient familiarity that they do not need to read verbatim from notes or slides. Memorizing a script is not required, but simply reading a prepared text aloud throughout the entire presentation is not acceptable and will be penalized. The presentation should be a genuine act of communication, not a recitation.
# 5. Critical Warnings
## The following issues will result in significant grade penalties:
Theoretical or modeling incoherences. A mismatch between the research question and the analytical approach — for instance, using a loss function incompatible with the task, applying a model architecture unsuited to the data, or performing regressions that do not align with the stated hypothesis — will be heavily penalized.
Falsified results. Any fabrication or manipulation of results will be treated as a serious academic integrity violation and penalized accordingly. The objective of this project is not to produce groundbreaking research but to demonstrate a rigorous application of the skills covered in this course. A well-reasoned project with modest or null results will receive a good grade, provided the methodology and interpretation are sound.
# 6. Submission
## All deliverables must be submitted on March 30 before midnight. Each group should provide:
The report in PDF format.
The complete codebase and data.
The presentation slides.
Submissions may be shared via any practical format that allows both the code and the data to be included — for example, a GitHub repository, a shared Google Drive folder, or a compressed archive. The key requirement is that the examiners must be able to access and review the full pipeline without difficulty.
# 7. Evaluation Summary
