<div align='center'>

# Assessing The Effect of Various Sampling Frequency and Double Soft F1-loss in Multi-Label Classification of 12-Lead ECGs  
  
[Bjørn-Jostein Singstad](https://github.com/Bsingstad)<br>
Department of Computational Physiology<br>
Simula Research Laboratory<br>
Oslo, Norway<br>
<b.j.singstad@fys.uio.no><br>

[Eraraya Morenzo Muten](https://github.com/morenzoe)<br>
Department of Biomedical Engineering<br>
Institut Teknologi Bandung<br>
Bandung, Indonesia<br>
<morenzomuten@ieee.org><br>
  
</div>
  
### Abstract

  The electrocardiogram (ECG) is an almost universally accessible diagnostic tool for heart disease. An ECG is measured by using an electrocardiograph, and today’s electrocardiographs use built-in software to interpret the ECGs automatically after they are recorded. However, these algorithms show limited performance, and the clinicians usually have to manually interpret the ECG, regardless of whether the ECG has been interpreted by an algorithm or not. Manual interpretation of the ECG can be time-consuming and require specific skills, and it’s therefore clearly a need for better interpretation algorithms to make ECG interpretations more accessible and time efficient. Algorithms based on artificial intelligence have shown promising performance in many fields, including ECG interpretation, over the last few years and might represent an alternative to manual ECG interpretation.

  In this study, we used a dataset with 88253 12-lead ECGs from multiple databases, annotated with SNOMED-CT codes by medical experts. We employed a supervised convolutional neural network with an Inception architecture to classify 30 of the most frequent annotated diagnoses in the dataset. Each patient could have more than one diagnosis at once, which makes it a multi-label classification. In this study, we compared the model’s performance using different preprocessing of the ECG and different model settings using 10-folded cross-validation. We compared the model’s classification performance using binary cross-entropy (BCE) loss and double soft F1 loss. Furthermore, we compared the classification performance while downsampling the original sampling rates of the input ECG. Finally, we trained a interpretable linear model   to provide class activation maps to explain the relative importance of each sample in the ECG with respect to the 30 diagnoses considered in this study.

  Due to the heavily imbalanced class distribution in our dataset, we chose to place the most emphasis on the F1 score when evaluating the performance of the models. Our results show that the best performance in terms of F1-score was seen when the Inception model used double soft F1 as the loss function and ECGs downsampled to 75Hz. This model achieved an F1 score of 0.420 ± 0.017, accuracy = 0.954 ± 0.002, and an AUROC score of 0.832 ± 0.019. An aggregation of the generated heatmaps, achieved using Local Interpretable Model-Agnostic Explanations (LIME), showed that the Inception model paid the most attention to the limb leads and the augmented leads and less importance to the precordial leads.

  One of the more significant contributions that emerge from this study is the use of aggregated heatmaps to obtain ECG lead importance for different diagnoses. In addition, we emphasized the relevance of evaluating different loss functions, and in this specific case, we found double soft F1 loss to be slightly better than BCE. Finally, we found it somewhat surprising that downsampling the ECG led to higher performance compared to the original 500Hz sampling rate. These findings contribute in several ways to our understanding of the artificial intelligence-based interpretation of ECGs, but further studies should be carried out in order to validate these findings on other cohorts as well.

Keywords ECG: Convolutional Neural Network, Multi-Label Classification, Explainable AI

### Declaration of conflicting interests
The author(s) declared no potential conflicts of interest concerning the research, authorship, and/or publication of this article.

### Funding
The author(s) disclosed receipt of the following financial support for the research, authorship, and/or publication of this article: This work was supported by the Norwegian Research Council (grant number: #309762 - ProCardio).

### Conclusion
The primary aim of the current study was to train a multi-label ECG classifier to achieve the best possible performance, given the unbalanced dataset score. Furthermore, we used this model to obtain class activation maps and based on those we found the leads that were considered the most important for each diagnosis. We also found that double soft F1-loss might improve the performance when classifying heavy imbalanced datasets. In addition, we observed that reducing the sampling frequency of the ECG from 500 Hz to around 100Hz   increased the model performance.
  
### References
[1]	[WHO Cardiovascular Diseases (CVDs)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

[2]	[S. Serge Barold. “Willem Einthoven and the birth of clinical electrocardiography a hundred years ago”](https://doi.org/10.1023/a:1023667812925)

[3]	[Martin Bickerton and Alison Pooler. “Misplaced ECG Electrodes and the Need for Continuing Training”](https://doi.org/10.12968/bjca.2019.14.3.123)

[4]	[Harold Smulyan. “The Computerized ECG: Friend and Foe”](https://doi.org/10.1016/j.amjmed.2018.08.025)

[5]	[Jürg Schläpfer and Hein J. Wellens. “Computer-Interpreted Electrocardiograms”](https://doi.org/10.1016/j.jacc.2017.07.723)

[6]	[Hassan Ismail Fawaz and et al. “Deep Learning for Time Series Classification: A Review”](https://doi.org/10.1007/s10618-019-00619-1)

[7]	[Antônio H. Ribeiro et al. “Automatic diagnosis of the 12-lead ECG using a deep neural network”](https://www.nature.com/articles/s41467-020-15432-4) 

[8]	[Jagadeeswara Rao Annam et al. “Classification of ECG Heartbeat Arrhythmia: A Review”](http://www.sciencedirect.com/science/article/pii/S1877050920310425)
 
[9]	[Qihang Yao et al. “Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network”](https://linkinghub.elsevier.com/retrieve/pii/S1566253518307632)

[10]	[Dengao Li et al. “Automatic Classification System of Arrhythmias Using 12-Lead ECGs with a Deep Neural Network Based on an Attention Mechanism”](https://www.mdpi.com/2073-8994/12/11/1827)

[11]	[Tsai-Min Chen et al. “Detection and Classification of Cardiac Arrhythmias by a Challenge-Best Deep Learning Neural Network Model”](https://linkinghub.elsevier.com/retrieve/pii/S2589004220300705)

[12]	[Zachi I Attia et al. “Application of artificial intelligence to the electrocardiogram”](https://doi.org/10.1093/eurheartj/ehab649)

[13]	[Awni Y. Hannun et al. “Cardiologist-level arrhythmia detection and classification in ambulatory electrocardio- grams using a deep neural network”](http://www.nature.com/articles/s41591-018-0268-3)

[14]	[Davide Castelvecchi. “Can we open the black box of AI?”](http://www.nature.com/news/can-we-open-the-black-box-of-ai-1.20731)

[15]	[Arun Rai. “Explainable AI: from black box to glass box”](https://doi.org/10.1007/s11747-019-00710-5)

[16]	[Ramprasaath R. Selvaraju et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization”](https://doi.org/10.1109/ICCV.2017.74)

[17]	[Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “"Why Should I Trust You?": Explaining the Predictions of Any Classifier”](http://arxiv.org/abs/1602.04938)

[18]	[Scott M Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model Predictions”](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

[19]	[van de Leur Rutger R. et al. “Automatic Triage of 12-Lead ECGs Using Deep Convolutional Neural Networks”](https://www.ahajournals.org/doi/10.1161/JAHA.119.015138)

[20]	[Steven A. Hicks and et al. “Explaining deep neural networks for knowledge discovery in electrocardiogram analysis”](https://www.nature.com/articles/s41598-021-90285-5)

[21]	[Dongdong Zhang et al. “Interpretable deep learning for automatic diagnosis of 12-lead electrocardiogram”](https://www.sciencedirect.com/science/article/pii/S2589004221003412)

[22]	[Inês Neves et al. “Interpretable heartbeat classification using local model-agnostic explanations on ECGs”](https://www.sciencedirect.com/science/article/pii/S0010482521001876)

[23]	[Ganeshkumar M. et al. “Explainable Deep Learning-Based Approach for Multilabel Classification of Electrocar- diogram”](https://doi.org/10.1109/TEM.2021.3104751)
 
[24]	[Apoorva Srivastava et al. “A deep residual inception network with channel attention modules for multi-label cardiac abnormality detection from reduced-lead ECG”](http://iopscience.iop.org/article/10.1088/1361-6579/ac6f40)

[25]	[Erick A. Perez Alday et al. “Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020”](https://dx.doi.org/10.1088/1361-6579/abc960)

[26]	[Matthew A Reyna et al. “Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021”](https://doi.org/10.23919/CinC53138.2021.9662687)

[27]	[Bjørn-Jostein Singstad and Christian Tronstad. “Convolutional Neural Network and Rule-Based Algorithms for Classifying 12-lead ECGs”](https://doi.org/10.22489/CinC.2020.227)

[28]	Bjørn-Jostein Singstad and Pål Haugar Brekke. “Multi-label ECG classification using Convolutional Neural Networks in a Classifier Chain”. In: Computing in Cardiology (2021). In Review.

[29]	[Feifei Liu et al. “An Open Access Database for Evaluating the Algorithms of Electrocardiogram Rhythm and Morphology Abnormality Detection”](https://doi.org/10.1166/jmihi.2018.2442)

[30]	Vikto Tihonenko et al. “St Petersburg INCART 12-lead arrhythmia database”. In: PhysioBank PhysioToolkit and PhysioNet (2008).

[31]	[Patrick Wagner et al. “PTB-XL, a Large Publicly Available Electrocardiography Dataset”](https://doi.org/10.1038/s41597-020-0495-6)

[32]	[R. Bousseljot, D. Kreiseler, and A. Schnabel. “Nutzung der EKG-Signaldatenbank Cardiodat der PTB über das Internet”](https://doi.org/10.1515/bmte.1995.40.s1.317)

[33]	[Jianwei Zheng and et al. “Optimal Multi-Stage Arrhythmia Classification Approach”](https://doi.org/10.1038/s41598-020-59821-7)

[34]	[Christian Szegedy et al. "Going Deeper with Convolutions"](http://arxiv.org/abs/1409.4842)

[35]	[Martín Abadi et al. “TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems”](http://arxiv.org/abs/1603.04467)

[36]	[David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. “Learning representations by back-propagating errors”](https://www.nature.com/articles/323533a0)

[37]	[Quintin van Lohuizen. “Training Deep Neural Networks with Soft Loss for Strong Gravitational Lens Detection”](https://www.ai.rug.nl/~mwiering/Thesis_Quintin_van_Lohuizen.pdf)

[38]	[Diederik P. Kingma and Jimmy Ba. “Adam: A Method for Stochastic Optimization”](http://arxiv.org/abs/1412.6980)

[39]	[Connor Shorten and Taghi M. Khoshgoftaar. “A survey on Image Data Augmentation for Deep Learning”](https://doi.org/10.1186/s40537-019-0197-0)

[40]	[Marzyeh Ghassemi, Luke Oakden-Rayner, and Andrew L Beam. “The false hope of current approaches to explainable artificial intelligence in health care”](https://www.sciencedirect.com/science/article/pii/S2589750021002089)

### License
  
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
