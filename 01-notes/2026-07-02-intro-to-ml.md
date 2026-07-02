# Introduction to Machine Learning — 2026-07-02

## TL;DR
Machine Learning is a set of techniques that let a program find patterns in data and make predictions or decisions, without being explicitly programmed with hardcoded rules for every case.

## Key concepts
- **Supervised learning**: model learns from labeled data (input → known output). Two types:
  - *Regression*: predicting a continuous number (e.g. house price)
  - *Classification*: predicting a category/class (e.g. spam vs not-spam)
- **Unsupervised learning**: no labels — model finds structure on its own (e.g. clustering customers into segments).
- **Reinforcement learning**: agent learns by interacting with an environment and getting rewards/penalties
- **Why ML over traditional programming**: traditional code needs explicit rules; ML works well when rules are too complex or unknown to hand-code (e.g. recognizing handwriting).
- **Train/test split**: always evaluate a model on data it hasn't seen, otherwise you're just measuring memorization, not generalization.
- **Features vs target**: features (X) are the inputs the model uses; target (y) is what you're trying to predict.

## Math / formula
No heavy math today — foundational/conceptual module. 

## Code
No coding today

## Interview-style question I could be asked
**Q: What's the difference between supervised and unsupervised learning? Give an example of each.**
A: Supervised learning trains on labeled data to predict a known outcome — e.g. predicting house prices from features like size and location. Unsupervised learning works on unlabeled data to find hidden structure — e.g. clustering customers into segments based on purchasing behavior, with no predefined "correct" grouping.

## Links
- https://www.coursera.org/learn/machine-learning-with-python