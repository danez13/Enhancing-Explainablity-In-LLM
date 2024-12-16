# CAP-XAI
## [documentation](https://danez13.github.io/Enhancing-Explainablity-In-LLM/)
## Prerequisites
1. python 11.0
2. install the requirements
```
pip install -r requirements.txt
```
3. download the glove embeddings
```
python XAI-setup.py
```
## Steps to Run
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

## Original Paper
* [Github](https://github.com/danez13/xai-benchmark)
* [Paper](https://arxiv.org/abs/2009.13295)

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

