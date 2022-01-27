# Synthetic ECG Generation using GAN Models, a Comparison
Electrocardiogram (ECG) datasets tend to be highly imbalanced due to the scarcity of abnormal cases. Additionally, the use of real patients' ECG is highly regulated due to privacy issues. Therefore, there is always a need for more ECG data, especially for the training of automatic diagnosis machine learning models, which perform better when trained on a balanced dataset. We studied the synthetic ECG generation capability of 5 different models from the generative adversarial network (GAN) family and compared their performances, the focus being only on Normal cardiac cycles. Dynamic Time Warping (DTW), Fréchet, and Euclidean distance functions were employed to quantitatively measure performance. Five different methods for evaluating generated beats were proposed and applied. We also proposed 3 new concepts (threshold, accepted beat and productivity rate) and employed them along with the aforementioned methods as a systematic way for comparison between models. The results show that all the tested models can to an extent successfully mass-generate acceptable heartbeats with high similarity in morphological features, and potentially all of them can be used to augment imbalanced datasets. However, visual inspections of generated beats favor BiLSTM-DC GAN and WGAN, as they produce statistically more acceptable beats. Also, with regards to productivity rate, the Classic GAN is superior with a 72% productivity rate.

Paper:
https://arxiv.org/abs/2112.03268

for citation please use:

@article{adib2021synthetic,
  title={Synthetic ECG Signal Generation Using Generative Neural Networks},  
  author={Adib, Edmond and Afghah, Fatemeh and Prevost, John J},  
  journal={arXiv preprint arXiv:2112.03268},  
  year={2021}
}
