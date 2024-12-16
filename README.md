# CAP-XAI
## Directory Structure
The code is broken up into various directories
* **Enhancing Explainability in LLM**
    * This overarching directory that contains the entire project
* **data**: a directory containing data that is used across the entire project in various modules
    * **analysis**: contains graphs and visuals for results of the project
    * **e-SNLI**: The dataset used across the entire project
    * **evaluations**: the data from the evaluation criteria from which the model was analysed
    * **models**: data containing model parameters and predictions
    * **saliency**: data containg generated saliency values from every XAI metric
* **eval_analysis**: contains analysis script that visualizes the evaluation criteria
* **models**: contains scripts and functions for data handling, model construction, saliency utilities, and model training
* **saliency_eval**: contains scripts and functions for evaluation the generated XAI metrics based on the criteria
* **saliency_gen**: contains scripts and functions for generating the XAI metric values based on the model predictions
* **WebSite**: contains HTML, CSS, and javascript code for generating a documentation website that explains the code is detail

## User Manual
### Prerequisites
1. python 11.0
2. install the requirements
```
pip install -r requirements.txt
```
3. download the glove embeddings
```
python XAI-setup.py
```
### How to Run
1. train the model
```
python -m models.train_cnn
```
2. generate XAI metrics for the model
```
python -m saliency_gen.interpret_grads_occ

python -m saliency_gen.interpret_lime

python -m saliency_gen.interpret_shap

python -m saliency_gen.generate_random_sal
```
3. evaluate the metrics
```
python -m saliency_eval.confidence

python -m saliency_eval.faithfulness

python -m saliency_eval.human_agreement

python -m saliency_eval.consistency_precompute
python -m saliency_eval.consistency_rats

python -m saliency_eval. consist_data_sample_instance_pairs
python -m saliency_eval.consist_data
```
4. analyze the evaluations
```
python eval_analysis/analysis.py
```

## Citations
Bastings, J., Ebert, S., Zablotskaia, P., Sandholm, A., &
Filippova, K. (2021). " Will You Find These Shortcuts?" A
Protocol for Evaluating the Faithfulness of Input Salience
Methods for Text Classification. arXiv preprint
arXiv:2111.07367.

Atanasova, P. (2024). A diagnostic study of explainability
techniques for text classification. In Accountable and
Explainable Methods for Complex Reasoning over Text (pp.
155-187). Cham: Springer Nature Switzerland.

Jain, S., & Wallace, B. C. (2019). Attention is not
explanation. arXiv preprint arXiv:1902.10186.

Zhou, Y., Booth, S., Ribeiro, M. T., & Shah, J. (2022, June).
Do feature attribution methods correctly attribute features?.
In Proceedings of the AAAI Conference on Artificial
Intelligence (Vol. 36, No. 9, pp. 9623-9633).

Wang, Zichong, et al. "History, development, and principles of large language models: an introductory survey." AI and Ethics (2024): 1-17.

Arrieta, Alejandro Barredo, et al. "Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI." Information fusion 58 (2020): 82-115.

Dwivedi, R., Dave, D., Naik, H., Singhal, S., Rana, O.F., Patel, P., Qian, B., Wen, Z., Shah, T., Morgan, G., & Ranjan, R. (2022). Explainable AI (XAI): Core Ideas, Techniques, and Solutions. ACM Computing Surveys, 55, 1 - 33.

Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., ... & Du, M. (2024). Explainability for large language models: A survey. ACM Transactions on Intelligent Systems and Technology, 15(2), 1-38.

