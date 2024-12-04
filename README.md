# CAP-XAI
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

python -m saliency_eval.consist_data_sample_instance_pairs
python -m saliency_eval.consist_data
```
