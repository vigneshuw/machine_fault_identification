# Real-time fault identification system for a retrofitted ultra-precision CNC machine from equipment's power consumption data: A case study of an implementation

This repository contains the code corresponding to the study published in IJPEM-GT. For more details please visit [MINLab](https://min.me.wisc.edu/) and/or [Smart Manufacturing at UW-Madison](https://smartmfg.me.wisc.edu/)

The real-time inference system developed from this study can be seen [here](https://smartmfg.me.wisc.edu/pages/dashboards/energy_monitoring/robonano1_enms.html) 

DOI Link:-


## Abstract

Ability to detect faults in manufacturing machines have become crucial in the era of Smart Manufacturing to enable cost savings from erratic downtimes, in an effort towards Green Manufacturing. The power consumption data provides myriad of information that would facilitate condition monitoring of manufacturing machines. In this work, we retrofit an ultra-precision CNC machine using an inexpensive power meter. The data collected from the power meter were streamed in real-time to Amazon Web Services (AWS) servers using industry standard Message Query Telemetry Transport (MQTT) protocol. The error identification study was carried out in two-folds, we first identify if the error has occurred followed by classifying the type of controller error. The study also develops anomaly detection models to identify normal operating condition of the machine from the anomalous error states. Anomaly detection was particularly favorable for manufacturing machines as it requires data only from the normal operating conditions of the machine. The developed models performed with macro F1-Score of 0.9971 $\pm$ 0.0012 and 0.9974 $\pm$ 0.0018 for binary and multiclass classification respectively. The anomaly detection models were able to identify the anomalous data instances with an average accuracy of 95%. A feature importance study was then carried out to identify the most valuable feature for error identification. Finally, the trained models were containerized and hosted at AWS. The overarching goal of this project was to develop a complete inexpensive ML pipeline that would enable industries to detect operation anomalies in manufacturing machines just from the energy consumption data of the machine.


## Directory Information

1. **anomaly_detection** -> Model development for the anomaly detection studies.
2. **aws-sagemaker** -> Scripts that help in hosting the developed models at the AWS.
3. **figures** -> Plots created from the analysis performed.
4. **lib** -> Libraries used for data processing, feature extraction, model development, and model evaluation, for both supervised and unsupervised learning.
5. **results** -> Results from model evaluation and the feature importance study conducted.
6. **testing** -> Scripts to test the hosted models.

